from typing import Dict

from functools import partial

import math
from typing import Callable

import jax
from chex import PRNGKey
from jax import Array, random
from jax import numpy as jnp

from ppomdp.core import (
    Carry,
    OuterState,
    InnerState,
    OuterParticles,
    TransitionModel,
    ObservationModel,
    RewardFn,
    RecurrentPolicy,
)


# smc functions
def resample_inner(
    rng_key: PRNGKey,
    inner_state: InnerState,
    resample_fn: Callable,
) -> InnerState:
    """Resample the inner particles for a single outer trajectory.

    Only resamples if the effective sample size is below 75% of the number of particles.

    Args:
        rng_key: The random number generator key.
        inner_state: The state associated with a single outer trajectory.
            Leaves have shape (M, ...).
        resample_fn: The resampling function.
    """
    num_particles = inner_state.particles.shape[0]

    def true_fn(state: InnerState) -> InnerState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        return InnerState(
            particles=state.particles[resampling_idx],
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
        )

    def false_fn(state: InnerState) -> InnerState:
        resampling_idx = jnp.arange(num_particles)
        return state._replace(resampling_indices=resampling_idx)

    resampled_state = jax.lax.cond(
        effective_sample_size(inner_state.weights) < 0.75 * num_particles,
        true_fn, false_fn, inner_state
    )
    return resampled_state


def propagate_inner(
    rng_key: PRNGKey,
    model: TransitionModel,
    particles: Array,
    action: Array,
) -> Array:
    r"""Propagate the inner particles for a single trajectory.

    Args:
        rng_key: PRNGKey
        model: TransitionModel
            The transition model for the state.
        particles: Array
            The state particles $\{s_{t-1}^{nm}\}_{m=1}^M$ associated with
            the n-th outer trajectory. Has shape (M, ...).
        action: Array
            The action $a_{t-1}^n$.
    """
    num_particles = particles.shape[0]
    rng_keys = random.split(rng_key, num_particles)
    return jax.vmap(model.sample, in_axes=(0, 0, None))(rng_keys, particles, action)


def reweight_inner(
    model: ObservationModel, state: InnerState, obs: Array
) -> InnerState:
    r"""Reweight the inner particles for a single outer trajectory.

    Args:
        model: ObservationModel
            The observation model.
        state: InnerState
            The inner state associated with the n-th outer trajectory.
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
    rng_key: PRNGKey, obs_model: ObservationModel, state: InnerState
) -> Array:
    r"""Sample from the marginal observation distribution.

    $z_t^n \sim \sum_{m=1}^M W_{s,t}^{nm} h(z_t \mid s_t^{nm})$.

    Args:
        rng_key: PRNGKey
        obs_model: ObservationModel
        state: InnerState
            The inner state associated with the n-th outer trajectory.
            Leaves have shape (M, ...).
    """
    key1, key2 = random.split(rng_key)
    x = random.choice(key1, state.particles, p=state.weights)
    return obs_model.sample(key2, x)


def expected_reward(
    reward_fn: RewardFn,
    inner_state: InnerState,
    action: Array,
    time_idx: int
) -> Array:
    """
    Calculate the expected reward for a given particle and inner state.

    Args:
        reward_fn: RewardFn
            The reward function, r(s_{t], a_{t-1}).
        inner_state: InnerState
            The inner state containing particles and weights.
        action: Array
            Action particle.
        time_idx: int
            The current time index.

    Returns:
        Array: The cumulative return for the given particle and inner state.
    """
    rewards = jax.vmap(reward_fn, in_axes=(0, None, None))(inner_state.particles, action, time_idx)
    return jnp.sum(rewards * inner_state.weights)


def log_potential(
    reward_fn: RewardFn,
    inner_state: InnerState,
    action: Array,
    prev_action: Array,
    time_idx: int,
    tempering: float,
    slew_rate_penalty: float,
) -> tuple[Array, Array]:
    r"""Estimate the log potential function.

    The estimate for the log potential function is given by
    .. math::
    \log g_t^n = \eta * \sum_{m=1}^M W_{s,t}^{nm} r_t(s_t^{nm}, a_t^n).

    Args:
        reward_fn: The reward function.
        inner_state: The inner state associated with the n-th outer trajectory.
            Leaves have shape (M, ...).
        action: The current action.
        prev_action: The previous action.
        time_idx: The current time index.
        tempering: The tempering parameter.
        slew_rate_penalty: The slew rate penalty.
    """
    rewards = expected_reward(reward_fn, inner_state, action, time_idx)
    mod_rewards = rewards - slew_rate_penalty * jnp.dot(action - prev_action, action - prev_action)
    return tempering * mod_rewards, rewards


def resample_outer(
    rng_key: PRNGKey,
    outer_state: OuterState,
    resample: bool,
    resample_fn: Callable,
    conditional: bool = False
) -> OuterState:
    num_particles = outer_state.weights.shape[0]

    def true_fn(state: OuterState) -> OuterState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        # set zeroth resampling index to zero if conditional resampling is enabled
        resampling_idx = jax.lax.cond(
            conditional,
            lambda _: resampling_idx.at[0].set(0),
            lambda _: resampling_idx,
            None
        )
        resampled_particles = jax.tree.map(lambda x: x[resampling_idx], state.particles)
        resampled_rewards = state.rewards[resampling_idx]
        return OuterState(
            particles=resampled_particles,
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
            rewards=resampled_rewards,
        )

    def false_fn(state: OuterState) -> OuterState:
        resampling_idx = jnp.arange(num_particles)
        return state._replace(resampling_indices=resampling_idx)

    predicate = resample and effective_sample_size(outer_state.weights) < 0.75 * num_particles
    resampled_state = jax.lax.cond(predicate, true_fn, false_fn, outer_state)
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
    return jnp.clip(idx, 0, n - 1).astype(jnp.int_)


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
    return idx


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
    log_probs = jax.vmap(obs_model.log_prob, in_axes=(None, 0))(
        observation, states
    )
    return jax.nn.logsumexp(log_probs) - jnp.log(num_particles)


def policy_logpdf(
    actions: Array,
    observations: Array,
    policy: RecurrentPolicy,
    params: Dict,
) -> Array:
    """
    Compute the logpdf of init and all future actions for a single trajectory
    """
    def body(carry, args):
        action, observation = args
        carry, log_prob_inc = policy.carry_and_log_prob(action, observation, carry, params)
        return carry, log_prob_inc

    _, log_probs = jax.lax.scan(body, policy.reset(1), (actions[1:], observations[:-1]))
    return log_probs


def transition_logpdf(
    states: Array,
    log_weights: Array,
    action: Array,
    future_states: Array,
    trans_model: TransitionModel,
    obs_model: ObservationModel
):
    """" Transition probability of the inner particles for a single trajectory
    The transition density is marginalized over the resmapling indices
    """

    def _log_transition(next_state, states, action):
        logpdfs = jax.vmap(trans_model.log_prob, in_axes=(None, 0, None))(next_state, states, action)
        return jax.nn.logsumexp(logpdfs + log_weights)

    log_transitions = jax.vmap(_log_transition, in_axes=(0, None, None))(
        future_states, states, action
    )
    return jnp.sum(log_transitions)


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
    return jnp.einsum('mh,m->h', particles, weights) / jnp.sum(weights)


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
    return jnp.einsum('mh,ml,m->hl', centered, centered, weights) / jnp.sum(weights)


@partial(jnp.vectorize, signature="(m)->()")
def effective_sample_size(weights: Array) -> Array:
    """Compute the effective sample size."""
    return 1.0 / jnp.sum(jnp.square(weights))


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
