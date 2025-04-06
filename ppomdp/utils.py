import math
from functools import partial
from typing import Callable, Dict

import jax
from flax.training.train_state import TrainState
from jax import Array, random
from jax import numpy as jnp

from ppomdp.core import (
    Carry,
    PRNGKey,
    BeliefState,
    HistoryState,
    TransitionModel,
    ObservationModel,
    RecurrentPolicy,
    RewardFn,
)
from ppomdp.envs.core import POMDPEnv


# smc functions
def resample_belief(
    rng_key: PRNGKey,
    belief_state: BeliefState,
    resample_fn: Callable,
) -> BeliefState:
    """Resample the belief particles for a single history trajectory.

    Only resamples if the effective sample size is below 75% of the number of particles.

    Args:
        rng_key: The random number generator key.
        belief_state: The state associated with a single history trajectory.
            Leaves have shape (M, ...).
        resample_fn: The resampling function.

    Returns:
        The new `BeliefState` after resampling.
    """
    num_particles = belief_state.particles.shape[0]

    def true_fn(state: BeliefState) -> BeliefState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        return BeliefState(
            particles=state.particles[resampling_idx],
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
        )

    def false_fn(state: BeliefState) -> BeliefState:
        resampling_idx = jnp.arange(num_particles, dtype=jnp.int32)
        return state._replace(resampling_indices=resampling_idx)

    resampled_state = jax.lax.cond(
        effective_sample_size(belief_state.log_weights) < 0.75 * num_particles,
        true_fn,
        false_fn,
        belief_state,
    )
    return resampled_state


def propagate_belief(
    rng_key: PRNGKey,
    model: TransitionModel,
    particles: Array,
    action: Array,
) -> Array:
    r"""Propagate the belief particles for a single trajectory.

    Args:
        rng_key: PRNGKey
        model: TransitionModel
            The transition model for the state.
        particles: Array
            The state particles $\{s_{t-1}^{nm}\}_{m=1}^M$ associated with
            the n-th history trajectory. Has shape (M, ...).
        action: Array
            The action $a_{t-1}^n$.
    """
    num_particles = particles.shape[0]
    rng_keys = random.split(rng_key, num_particles)
    return jax.vmap(model.sample, in_axes=(0, 0, None))(rng_keys, particles, action)


def reweight_belief(
    model: ObservationModel, state: BeliefState, obs: Array
) -> BeliefState:
    r"""Reweight the belief particles for a single history trajectory.

    Args:
        model: ObservationModel
            The observation model.
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).
        obs: Array.
            The observation $z_t^n$.
    """
    log_weights = jax.vmap(model.log_prob, in_axes=(None, 0))(obs, state.particles)
    log_weights += state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    return state._replace(log_weights=log_weights, weights=weights)


def sample_marginal_obs(
    rng_key: PRNGKey, obs_model: ObservationModel, state: BeliefState
) -> Array:
    r"""Sample from the marginal observation distribution.

    $z_t^n \sim \sum_{m=1}^M W_{s,t}^{nm} h(z_t \mid s_t^{nm})$.

    Args:
        rng_key: PRNGKey
        obs_model: ObservationModel
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).
    """
    key1, key2 = random.split(rng_key)
    x = random.choice(key1, state.particles, p=state.weights)
    return obs_model.sample(key2, x)


def expected_reward(
    belief_state: BeliefState, action: Array, time_idx: int, reward_fn: RewardFn
) -> Array:
    """
    Calculate the expected reward for a given particle and belief state.

    Args:
        reward_fn: RewardFn
            The reward function, r(s_{t], a_{t-1}).
        belief_state: BeliefState
            The belief state containing particles and weights.
        action: Array
            Action particle.
        time_idx: int
            The current time index.

    Returns:
        Array: The cumulative return for the given particle and belief state.
    """
    rewards = jax.vmap(reward_fn, in_axes=(0, None, None))(
        belief_state.particles, action, time_idx
    )
    if rewards.ndim != 1:
        raise ValueError("The reward function must return a scalar value.")
    return jnp.sum(rewards * belief_state.weights)


def log_potential(
    belief_state: BeliefState,
    action: Array,
    prev_action: Array,
    time_idx: int,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
) -> tuple[Array, Array]:
    r"""Estimate the log potential function.

    The estimate for the log potential function is given by
    .. math::
    \log g_t^n = \eta * \sum_{m=1}^M W_{s,t}^{nm} r_t(s_t^{nm}, a_t^n).

    Args:
        reward_fn: The reward function.
        belief_state: The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).
        action: The current action.
        prev_action: The previous action.
        time_idx: The current time index.
        tempering: The tempering parameter.
        slew_rate_penalty: The slew rate penalty.
    """
    rewards = expected_reward(belief_state, action, time_idx, reward_fn)
    mod_rewards = rewards - slew_rate_penalty * jnp.dot(
        action - prev_action, action - prev_action
    )
    return tempering * mod_rewards, rewards


def resample_history(
    rng_key: PRNGKey,
    history_state: HistoryState,
    resample_fn: Callable,
    conditional: bool = False,
) -> HistoryState:
    num_particles = history_state.weights.shape[0]

    def true_fn(state: HistoryState) -> HistoryState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        # Set zeroth resampling index to zero if conditional resampling is enabled
        resampling_idx = jax.lax.select(
            conditional, resampling_idx.at[0].set(0), resampling_idx
        )
        resampled_particles = jax.tree.map(lambda x: x[resampling_idx], state.particles)
        resampled_rewards = state.rewards[resampling_idx]
        return HistoryState(
            particles=resampled_particles,
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
            rewards=resampled_rewards,
        )

    def false_fn(state: HistoryState) -> HistoryState:
        resampling_idx = jnp.arange(num_particles, dtype=jnp.int32)
        return state._replace(resampling_indices=resampling_idx)

    predicate = effective_sample_size(history_state.log_weights) < 0.75 * num_particles
    resampled_state = jax.lax.cond(predicate, true_fn, false_fn, history_state)
    return resampled_state


def systematic_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    """
    Perform systematic resampling of particles based on their weights.

    Args:
        rng_key (PRNGKey): The random key for sampling.
        weights (Array): The weights of the particles.
        num_samples (int): The number of samples to draw.

    Returns:
        Array: The indices of the resampled particles.
    """
    n = weights.shape[0]
    u = random.uniform(rng_key, ())
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1).astype(jnp.int32)


def multinomial_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    """
    Perform multinomial resampling of particles based on their weights.

    Args:
        rng_key (PRNGKey): The random key for sampling.
        weights (Array): The weights of the particles.
        num_samples (int): The number of samples to draw.

    Returns:
        Array: The indices of the resampled particles.
    """
    idx = random.choice(rng_key, num_samples, shape=(num_samples,), p=weights)
    return idx.astype(jnp.int32)


# backward sampling functions
def marginal_observation_logpdf(
    states: Array,
    observation: Array,
    obs_model: ObservationModel,
):
    """
    Compute the log marginal likelihood of a single observation.

    Args:
        states: Array
            State particles.
        observation: Array
            The observation.
        obs_model: ObservationModel
            The observation model.
    """
    num_particles = states.shape[0]
    log_probs = jax.vmap(obs_model.log_prob, in_axes=(None, 0))(observation, states)
    return jax.nn.logsumexp(log_probs) - jnp.log(num_particles)


def policy_logpdf(
    time_idx: int,
    future_actions: Array,
    future_observations: Array,
    init_carry: list[Carry],
    init_observation: Array,
    policy: RecurrentPolicy,
    params: Dict,
) -> Array:
    """
    Compute the logpdf of init and all future actions for a single trajectory
    """
    num_steps = future_actions.shape[0]

    def body(k, val):
        carry, log_prob = val
        next_carry, log_prob_inc = policy.carry_and_log_prob(
            future_actions[k], carry, future_observations[k - 1], params
        )
        return next_carry, log_prob + log_prob_inc

    carry, log_prob = policy.carry_and_log_prob(
        future_actions[time_idx], init_carry, init_observation, params
    )

    _, log_prob = jax.lax.fori_loop(time_idx + 1, num_steps, body, (carry, log_prob))
    return log_prob


def transition_logpdf(
    future_belief_state: BeliefState,
    belief_state: BeliefState,
    action: Array,
    trans_model: TransitionModel,
):
    """Computes the transition probability of the belief particles for a single trajectory.

    The transition density is marginalized over the resampling indices.
    """

    def _log_transition(next_state, states, action):
        logpdfs = jax.vmap(trans_model.log_prob, in_axes=(None, 0, None))(
            next_state, states, action
        )
        return jax.nn.logsumexp(logpdfs + belief_state.log_weights)

    log_transitions = jax.vmap(_log_transition, in_axes=(0, None, None))(
        future_belief_state.particles, belief_state.particles, action
    )
    return jax.nn.logsumexp(log_transitions + future_belief_state.log_weights)


# misc functions
@partial(jnp.vectorize, signature="(m,h),(m)->(h)")
def weighted_mean(particles: Array, weights: Array):
    """
    Compute the weighted mean of the particles.

    Args:
        particles (Array): The particles array of shape (m, h), where m is the number of particles and h is the dimension.
        weights (Array): The weights array of shape (m,).

    Returns:
        Array: The weighted mean of the particles of shape (h,).
    """
    return jnp.einsum("mh,m->h", particles, weights) / jnp.sum(weights)


@partial(jnp.vectorize, signature="(m,h),(m)->(h,h)")
def weighted_covar(particles: Array, weights: Array) -> Array:
    """
    Compute the weighted empirical covariance of the particles.

    Args:
        particles (Array): The particles array of shape (m, h), where m is the number of particles and h is the dimension.
        weights (Array): The weights array of shape (m,).

    Returns:
        Array: The weighted covariance matrix of shape (h, h).
    """
    centered = particles - weighted_mean(particles, weights)
    return jnp.einsum("mh,ml,m->hl", centered, centered, weights) / jnp.sum(weights)


def log_ess(log_weights: Array) -> Array:
    """Computes the log of the effective sample size.

    Args:
        log_weights: Log-weights of particles.

    Returns:
        The logarithm of the effective sample size.
    """
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


@partial(jnp.vectorize, signature="(m)->()")
def effective_sample_size(log_weights: Array) -> Array:
    """Computes the effective sample size.

    Args:
        log_weights: Log-weights of particles.

    Returns:
        The effective sample size.
    """
    return jnp.exp(log_ess(log_weights))


@partial(jax.jit, static_argnames=("data_size", "batch_size", "skip_last"))
def batch_data(
    rng_key: Array,
    data_size: int,
    batch_size: int,
    skip_last: bool = True,
) -> list[Array]:
    """Generates batched indices.

    Args:
        rng_key (Array): Random key for shuffling the data.
        data_size (int): The size of the dataset to be batched.
        batch_size (int): The size of each batch.
        skip_last (bool, optional): If True, skips the last incomplete batch. Defaults to False.

    Returns:
        list[Array]: A list of batched indices.
    """
    batch_idx = random.permutation(rng_key, data_size)

    if skip_last:
        # Skip incomplete batch
        num_batches = data_size // batch_size
        batch_idx = batch_idx[: num_batches * batch_size]
    else:
        # Include incomplete batch
        num_batches = math.ceil(data_size / batch_size)

    batch_idx = jnp.array_split(batch_idx, num_batches)
    return batch_idx


def stitch_arrays(a: Array, b: Array, stitch: int, max_size: int):
    """
    Stitch two arrays `a` and `b` along the first axis up to a specified index.

    This function creates a new array by selecting elements from `a` up to the
    `stitch` index and elements from `b` for the remaining indices up to `max_size`.

    Args:
        a (Array): The first input array.
        b (Array): The second input array.
        stitch (int): The index up to which elements are taken from `a`.
        max_size (int): The total size of the output array.

    Returns:
        Array: The stitched array with elements from `a` up to `stitch` index
               and elements from `b` for the remaining indices.
    """

    return jnp.where(jnp.arange(max_size)[:, None] <= stitch, a, b)


def custom_split(rng_key: PRNGKey, num: int):
    """
    Splits a random number generator key into multiple sub-keys.

    Args:
        rng_key (PRNGKey): The random number generator key to split.
        num (int): The number of sub-keys to generate.

    Returns:
        tuple: A tuple containing the next key and an array of sub-keys.
    """
    key, *sub_keys = random.split(rng_key, num)
    return key, jnp.array(sub_keys)


@partial(jax.jit, static_argnames=("env_obj", "policy", "num_samples"))
def policy_evaluation(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy: RecurrentPolicy,
    train_state: TrainState,
    num_samples:int = 100
):
    """
    Deploy the (deterministic) policy to sample trajectories and evaluate the average reward.

    Args:
        rng_key (PRNGKey): The random number generator key.
        env_obj (POMDPEnv): The partially observable Markov decision process environment.
        policy (RecurrentPolicy): The policy to be evaluated.
        train_state (TrainState): The training state containing the policy parameters.
        num_samples (int, optional): The number of samples to draw. Defaults to 100.

    Returns:
        tuple: A tuple containing the expected reward, states, and actions.
    """
    eval_params = train_state.params.copy()
    eval_params["log_std"] = -20.0 * jnp.ones_like(eval_params["log_std"])

    def body(carry, key):
        states, policy_carry, observations, time_idx = carry

        # Sample actions.
        key, action_key = random.split(key)
        policy_carry, actions = policy.sample(action_key, policy_carry, observations, eval_params)

        # Sample next states.
        key, state_keys = custom_split(key, num_samples + 1)
        states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

        # Compute rewards.
        rewards = jax.vmap(env_obj.reward_fn, (0, 0, None))(states, actions, time_idx)

        # Sample observations.
        obs_keys = random.split(key, num_samples)
        observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

        return (states, policy_carry, observations, time_idx + 1), (states, actions, rewards)

    key, state_key = random.split(rng_key)
    init_states = env_obj.prior_dist.sample(seed=state_key, sample_shape=num_samples)
    key, obs_keys = custom_split(key, num_samples + 1)
    init_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, init_states)
    init_policy_carry = policy.reset(num_samples)

    _, (states, actions, rewards) = jax.lax.scan(
        body,
        (init_states, init_policy_carry, init_observations, 1),
        random.split(key, env_obj.num_time_steps + 1),
    )
    states = jnp.concatenate([init_states[None], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))
    return expected_reward, states, actions
