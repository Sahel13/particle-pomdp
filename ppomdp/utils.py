import math
from functools import partial
from typing import Callable, Dict

import jax
from jax import Array, random
from jax import numpy as jnp

from ppomdp.core import (
    BeliefState,
    Carry,
    HistoryParticles,
    HistoryState,
    ObservationModel,
    Parameters,
    PRNGKey,
    RecurrentPolicy,
    RewardFn,
    TransitionModel,
)
from ppomdp.envs.core import POMDPEnv


# smc functions
def resample_belief(
    rng_key: PRNGKey,
    belief_state: BeliefState,
    resample_fn: Callable,
) -> BeliefState:
    r"""Resample the belief particles for a single history trajectory using adaptive resampling.

    This function implements adaptive resampling - it only triggers resampling when
    the effective sample size (ESS) falls below 75% of the total number of particles.
    This helps prevent particle degeneracy while avoiding unnecessary resampling.

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling operations
        belief_state: BeliefState
            Current belief state containing:
            - particles: State particles with shape (M, ...)
            - weights: Normalized importance weights
            - log_weights: Log of importance weights
            - resampling_indices: Indices from previous resampling
        resample_fn: Callable
            Function implementing the resampling strategy (e.g. systematic, multinomial)

    Returns:
        BeliefState:
            Updated belief state after resampling (if triggered) with:
            - Resampled particles
            - Reset weights to uniform
            - Recorded resampling indices

    Notes:
        - If ESS > 0.75M, returns original state with only indices updated
        - If ESS ≤ 0.75M, performs resampling and resets weights
        - The effective sample size is calculated as:
          $ESS = \frac{(\sum_{m=1}^M w_m)^2}{\sum_{m=1}^M w_m^2}$
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
        effective_sample_size(belief_state.log_weights) < 0.5 * num_particles,
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
    r"""Propagate belief particles forward using the transition model.

    This function advances each particle through the state transition model
    to get the next state distribution. It handles vectorized operations across
    all particles efficiently.

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling
        model: TransitionModel
            The transition model for the state.
        particles: Array
            The state particles $\{s_{t-1}^{nm}\}_{m=1}^M$ associated with
            the n-th history trajectory. Has shape (M, ...).
        action: Array
            The action $a_{t-1}^n$.

    Returns:
        Array:
            The propagated particles $\{s_t^{nm}\}_{m=1}^M$ with shape (M, ...).
    """
    num_particles = particles.shape[0]
    rng_keys = random.split(rng_key, num_particles)
    return jax.vmap(model.sample, in_axes=(0, 0, None))(rng_keys, particles, action)


def reweight_belief(
    model: ObservationModel,
    state: BeliefState,
    observation: Array
) -> BeliefState:
    r"""Update particle weights based on new observation.

    Computes importance weights for particles by evaluating the observation
    likelihood. Weights are normalized to sum to 1.

    Args:
        model: ObservationModel
            The observation model.
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).
        observation: Array
            The observation $z_t^n$.

    Returns:
        BeliefState:
            Updated belief state with new weights computed as:
            $w_t^{nm} = \frac{w_{t-1}^{nm} p(z_t^n | s_t^{nm})}{\sum_{m'=1}^M w_{t-1}^{nm'} p(z_t^n | s_t^{nm'})}$
    """
    log_weights = jax.vmap(model.log_prob, in_axes=(None, 0))(observation, state.particles)
    log_weights += state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    return state._replace(log_weights=log_weights, weights=weights)


def sample_marginal_obs(
    rng_key: PRNGKey,
    obs_model: ObservationModel,
    state: BeliefState
) -> Array:
    r"""Sample an observation from the marginal observation distribution.

    Samples an observation from the weighted mixture of observation models:
    $z_t^n \sim \sum_{m=1}^M W_{s,t}^{nm} h(z_t \mid s_t^{nm})$

    Args:
        rng_key: PRNGKey
            Random number generator key
        obs_model: ObservationModel
            The observation model
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).

    Returns:
        Array:
            A sampled observation from the marginal distribution
    """
    key, sub_key = random.split(rng_key)
    x = random.choice(sub_key, state.particles, p=state.weights)
    return obs_model.sample(key, x)


def log_marginal_obs(
    obs_model: ObservationModel,
    observation: Array,
    state: BeliefState
) -> Array:
    r"""Sample an observation from the marginal observation distribution.

    Samples an observation from the weighted mixture of observation models:
    $z_t^n \sim \sum_{m=1}^M W_{s,t}^{nm} h(z_t \mid s_t^{nm})$

    Args:
        obs_model: ObservationModel
            The observation model
        observation:
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).

    Returns:
        Array:
            A sampled observation from the marginal distribution
    """
    log_probs = jax.vmap(obs_model.log_prob, in_axes=(None, 0))(observation, state.particles)
    return jax.nn.logsumexp(log_probs + state.log_weights) - jax.nn.logsumexp(state.log_weights)


def expected_reward(
    belief_state: BeliefState,
    action: Array,
    time_idx: int,
    reward_fn: RewardFn
) -> Array:
    r"""Calculate the expected reward for a given particle and belief state.

    Computes the expected reward by taking a weighted average over the belief state:
    $\mathbb{E}[r_t] = \sum_{m=1}^M w_t^{nm} r(s_t^{nm}, a_t^n)$

    Args:
        belief_state: BeliefState
            The belief state containing particles and weights
        action: Array
            Action particle
        time_idx: int
            The current time index
        reward_fn: RewardFn
            The reward function, r(s_{t}, a_{t-1})

    Returns:
        Array:
            The expected reward for the given particle and belief state
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
    slew_rate_penalty: float,
    tempering: float,
) -> tuple[Array, Array]:
    r"""Estimate the log potential function.

    The estimate for the log potential function is given by:
    $\log g_t^n = \eta \left( \sum_{m=1}^M W_{s,t}^{nm} r_t(s_t^{nm}, a_t^n) - \lambda \|a_t^n - a_{t-1}^n\|^2 \right)$

    Args:
        belief_state: BeliefState
            The belief state associated with the n-th history trajectory
        action: Array
            The current action
        prev_action: Array
            The previous action
        time_idx: int
            The current time index
        reward_fn: RewardFn
            The reward function
        slew_rate_penalty: float
            The penalty coefficient for action changes
        tempering: float
            The tempering parameter η

    Returns:
        tuple[Array, Array]:
            - The tempered log potential value
            - The raw reward value before tempering
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
    r"""Resample history particles using adaptive resampling.

    Performs adaptive resampling on the history state, similar to resample_belief,
    but operates on the full history trajectory. Can optionally perform conditional
    resampling where the first particle is preserved.

    Args:
        rng_key: PRNGKey
            Random number generator key
        history_state: HistoryState
            Current history state containing particles, weights, and rewards
        resample_fn: Callable
            Function implementing the resampling strategy
        conditional: bool, optional
            If True, preserves the first particle during resampling

    Returns:
        HistoryState:
            Resampled history state with updated particles and uniform weights
    """
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

    predicate = effective_sample_size(history_state.log_weights) < 0.5 * num_particles
    resampled_state = jax.lax.cond(predicate, true_fn, false_fn, history_state)
    return resampled_state


def systematic_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    r"""Perform systematic resampling of particles based on their weights.

    Implements systematic resampling which provides lower variance than multinomial
    resampling. The algorithm generates a single uniform random number and uses it
    to create a deterministic grid of points for resampling.

    Args:
        rng_key: PRNGKey
            Random number generator key
        weights: Array
            The weights of the particles
        num_samples: int
            The number of samples to draw

    Returns:
        Array:
            The indices of the resampled particles

    Notes:
        The systematic resampling algorithm:
        1. Generate a single uniform random number u ~ U[0,1]
        2. Create a grid of points: u_i = (i + u)/N for i = 0,...,N-1
        3. For each u_i, find the particle whose cumulative weight interval contains u_i
    """
    n = weights.shape[0]
    u = random.uniform(rng_key, ())
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1).astype(jnp.int32)


def multinomial_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    r"""Perform multinomial resampling of particles based on their weights.

    Implements multinomial resampling where each particle is sampled independently
    according to its weight. This is simpler but has higher variance than systematic
    resampling.

    Args:
        rng_key: PRNGKey
            Random number generator key
        weights: Array
            The weights of the particles
        num_samples: int
            The number of samples to draw

    Returns:
        Array:
            The indices of the resampled particles

    Notes:
        The multinomial resampling algorithm:
        1. Generate N independent uniform random numbers
        2. For each random number, find the particle whose cumulative weight interval contains it
    """
    idx = random.choice(rng_key, num_samples, shape=(num_samples,), p=weights)
    return idx.astype(jnp.int32)


# backward sampling functions
def marginal_observation_logpdf(
    states: Array,
    observation: Array,
    obs_model: ObservationModel,
):
    r"""Compute the log marginal likelihood of a single observation.

    Calculates the log probability of an observation by marginalizing over the state particles:
    $\log p(z_t^n) = \log \left( \frac{1}{M} \sum_{m=1}^M p(z_t^n | s_t^{nm}) \right)$

    Args:
        states: Array
            State particles
        observation: Array
            The observation
        obs_model: ObservationModel
            The observation model

    Returns:
        Array:
            The log marginal likelihood of the observation
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
    r"""Compute the logpdf of a sequence of actions under the policy.

    Calculates the log probability of a sequence of actions given observations:
    $\log p(a_{t:T}^n | z_{t:T}^n) = \sum_{k=t}^T \log p(a_k^n | a_{1:k-1}^n, z_{1:k}^n)$

    Args:
        time_idx: int
            The starting time index
        future_actions: Array
            The sequence of future actions
        future_observations: Array
            The sequence of future observations
        init_carry: list[Carry]
            Initial policy carry state
        init_observation: Array
            Initial observation
        policy: RecurrentPolicy
            The policy to evaluate
        params: Dict
            Policy parameters

    Returns:
        Array:
            The log probability of the action sequence
    """
    num_steps = future_actions.shape[0]

    def body(k, val):
        carry, log_prob = val
        next_carry, log_prob_inc = policy.carry_and_log_prob(
            next_actions=future_actions[k],
            carry=carry,
            actions=None,
            observations=future_observations[k - 1],
            params=params,
        )
        return next_carry, log_prob + log_prob_inc

    carry, log_prob = policy.carry_and_log_prob(
        next_actions=future_actions[time_idx],
        carry=init_carry,
        actions=None,
        observations=init_observation,
        params=params,
    )

    _, log_prob = jax.lax.fori_loop(time_idx + 1, num_steps, body, (carry, log_prob))
    return log_prob


def transition_logpdf(
    future_belief_state: BeliefState,
    belief_state: BeliefState,
    action: Array,
    trans_model: TransitionModel,
):
    """Compute the transition probability between belief states."""

    def log_transition_single(next_state, states):
        """Compute the transition probability to a single `next_state`.

        This is marginalized over the resampling indices (assuming multinomial resampling).
        """
        logpdfs = jax.vmap(trans_model.log_prob, in_axes=(None, 0, None))(
            next_state, states, action
        )
        return jax.nn.logsumexp(logpdfs + belief_state.log_weights)

    log_transitions = jax.vmap(log_transition_single, in_axes=(0, None))(
        future_belief_state.particles, belief_state.particles
    )
    return jnp.sum(log_transitions)


# misc functions
@partial(jnp.vectorize, signature="(m,h),(m)->(h)")
def weighted_mean(particles: Array, weights: Array):
    r"""Compute the weighted mean of the particles.

    Calculates the weighted mean of particles:
    $\mu = \frac{\sum_{m=1}^M w_m x_m}{\sum_{m=1}^M w_m}$

    Args:
        particles: Array
            The particles array of shape (m, h), where m is the number of particles and h is the dimension
        weights: Array
            The weights array of shape (m,)

    Returns:
        Array:
            The weighted mean of the particles of shape (h,)
    """
    return jnp.einsum("mh,m->h", particles, weights) / jnp.sum(weights)


@partial(jnp.vectorize, signature="(m,h),(m)->(h,h)")
def weighted_covar(particles: Array, weights: Array) -> Array:
    r"""Compute the weighted empirical covariance of the particles.

    Calculates the weighted covariance matrix:
    $\Sigma = \frac{\sum_{m=1}^M w_m (x_m - \mu)(x_m - \mu)^T}{\sum_{m=1}^M w_m}$

    Args:
        particles: Array
            The particles array of shape (m, h), where m is the number of particles and h is the dimension
        weights: Array
            The weights array of shape (m,)

    Returns:
        Array:
            The weighted covariance matrix of shape (h, h)
    """
    centered = particles - weighted_mean(particles, weights)
    return jnp.einsum("mh,ml,m->hl", centered, centered, weights) / jnp.sum(weights)


def log_ess(log_weights: Array) -> Array:
    r"""Computes the log of the effective sample size.

    Calculates the log of the effective sample size (ESS):
    $\log(ESS) = 2\log\left(\sum_{m=1}^M w_m\right) - \log\left(\sum_{m=1}^M w_m^2\right)$

    Args:
        log_weights: Array
            Log-weights of particles

    Returns:
        Array:
            The logarithm of the effective sample size
    """
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


@partial(jnp.vectorize, signature="(m)->()")
def effective_sample_size(log_weights: Array) -> Array:
    r"""Computes the effective sample size.

    Calculates the effective sample size (ESS):
    $ESS = \frac{(\sum_{m=1}^M w_m)^2}{\sum_{m=1}^M w_m^2}$

    Args:
        log_weights: Array
            Log-weights of particles

    Returns:
        Array:
            The effective sample size
    """
    return jnp.exp(log_ess(log_weights))


@partial(jax.jit, static_argnames=("data_size", "batch_size", "skip_last"))
def batch_data(
    rng_key: Array,
    data_size: int,
    batch_size: int,
    skip_last: bool = True,
) -> list[Array]:
    r"""Generates batched indices for data processing.

    Creates batches of indices for processing data in mini-batches. Can optionally
    skip the last incomplete batch.

    Args:
        rng_key: Array
            Random key for shuffling the data
        data_size: int
            The size of the dataset to be batched
        batch_size: int
            The size of each batch
        skip_last: bool, optional
            If True, skips the last incomplete batch. Defaults to True

    Returns:
        list[Array]:
            A list of batched indices

    Notes:
        - If skip_last=True, the last batch is dropped if it's smaller than batch_size
        - The indices are randomly permuted before batching
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
    r"""Stitch two arrays along the first axis up to a specified index.

    Creates a new array by selecting elements from `a` up to the `stitch` index
    and elements from `b` for the remaining indices up to `max_size`.

    Args:
        a: Array
            The first input array
        b: Array
            The second input array
        stitch: int
            The index up to which elements are taken from `a`
        max_size: int
            The total size of the output array

    Returns:
        Array:
            The stitched array with elements from `a` up to `stitch` index
            and elements from `b` for the remaining indices
    """

    return jnp.where(jnp.arange(max_size)[:, None] <= stitch, a, b)


def custom_split(rng_key: PRNGKey, num: int):
    r"""Splits a random number generator key into multiple sub-keys.

    Args:
        rng_key: PRNGKey
            The random number generator key to split
        num: int
            The number of sub-keys to generate

    Returns:
        tuple:
            A tuple containing the next key and an array of sub-keys
    """
    key, *sub_keys = random.split(rng_key, num)
    return key, jnp.array(sub_keys)


@jax.jit
def flatten_trajectories(particles: HistoryParticles):
    r"""Aligns particle trajectories in time and concatenates them to flatten the time dimension.

    Args:
        particles: HistoryParticles
            The particle trajectories to be flattened.
            Must include a time component with shape (num_time_steps, batch_size, ...)

    Returns:
        HistoryParticles:
            The flattened particle trajectories with the time dimension concatenated

    Raises:
        ValueError:
            If particles do not include a time component
    """

    if particles.observations.ndim != 3:
        raise ValueError("`particles` must include a time component.")

    actions = particles.actions[:-1].reshape((-1, particles.actions.shape[-1]))
    next_actions = particles.actions[1:].reshape((-1, particles.actions.shape[-1]))
    observations = particles.observations[:-1].reshape((-1, particles.observations.shape[-1]))
    next_observations = particles.observations[1:].reshape((-1, particles.observations.shape[-1]))
    carry = jax.tree.map(lambda x: x[:-1].reshape((-1, x.shape[-1])), particles.carry)
    return actions, next_actions, observations, next_observations, carry


@partial(jax.jit, static_argnames=("env_obj", "policy", "num_samples"))
def policy_evaluation(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy: RecurrentPolicy,
    params: Parameters,
    num_samples: int = 1024,
):
    r"""Deploy the (deterministic) policy to sample trajectories and evaluate the average reward.

    Args:
        rng_key: PRNGKey
            The random number generator key
        env_obj: POMDPEnv
            The partially observable Markov decision process environment
        policy: RecurrentPolicy
            The policy to be evaluated
        params: Parameters
            Stochastic policy parameters
        num_samples: int, optional
            The number of samples to draw. Defaults to 1024

    Returns:
        tuple:
            A tuple containing:
            - The expected reward
            - The sequence of states
            - The sequence of actions
    """

    def body(args, key):
        states, actions, carry, observations, time_idx = args

        # Sample actions.
        key, action_key = random.split(key)
        carry, _, actions = \
            policy.sample(action_key, carry, actions, observations, params)

        # Sample next states.
        key, state_keys = custom_split(key, num_samples + 1)
        states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

        # Compute rewards.
        rewards = jax.vmap(env_obj.reward_fn, (0, 0, None))(states, actions, time_idx)

        # Sample observations.
        obs_keys = random.split(key, num_samples)
        observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

        return (states,  actions, carry, observations, time_idx + 1), \
            (states, actions, rewards)

    key, state_key = random.split(rng_key)
    init_states = env_obj.prior_dist.sample(seed=state_key, sample_shape=num_samples)

    key, obs_keys = custom_split(key, num_samples + 1)
    init_carry = policy.reset(num_samples)
    init_actions = jnp.zeros((num_samples, policy.dim))
    init_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, init_states)

    _, (states, actions, rewards) = jax.lax.scan(
        f=body,
        init=(init_states, init_actions, init_carry, init_observations, 1),
        xs=random.split(key, env_obj.num_time_steps),
    )
    states = jnp.concatenate([init_states[None], states], axis=0)
    average_reward = jnp.mean(jnp.sum(rewards, axis=0))
    return average_reward, states, actions


def damping_schedule(
    step: int,
    total_steps: int,
    init_value: float = 0.1,
    max_value: float = 0.95,
    steepness: float = 1.0,
) -> float:
    r"""Generate smooth damping factor that increases over training.

    Uses a sigmoid schedule to gradually increase damping from init_value to max_value:
    $\beta = \frac{1}{1 + e^{-\alpha x}}$
    where x is a scaled time variable and α is the steepness parameter.

    Args:
        step: int
            Current training step
        total_steps: int
            Total number of training steps
        steepness: float, optional
            Controls slope of sigmoid curve. Defaults to 1.0
        init_value: float, optional
            Initial damping value. Defaults to 0.1
        max_value: float, optional
            Maximum damping value. Defaults to 0.95

    Returns:
        float:
            Damping factor between init_value and max_value
    """
    # Scale step to range roughly [-6, 6] for sigmoid input
    # This makes the steepest change happen around the middle of training
    x = (step / total_steps) * 12.0 - 6.0

    # Calculate sigmoid
    beta = 1 / (1 + jnp.exp(-steepness * x))

    # Rescale beta from [0, 1] range to [init_value, max_value] range
    beta_scaled = init_value + (max_value - init_value) * beta

    # Ensure value does not exceed max_value (due to potential float issues)
    # Note: If init_value < max_value, beta is <= 1, this might seem redundant,
    # but adds robustness.
    return jnp.minimum(beta_scaled, max_value)
