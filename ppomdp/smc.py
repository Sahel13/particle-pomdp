from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array, random

from chex import PRNGKey

from ppomdp.core import (
    LSTMCarry,
    InnerState,
    OuterState,
    TransitionModel,
    ObservationModel,
    RecurrentPolicy,
    RewardFn,
)


def resample_inner(rng_key: PRNGKey, state: InnerState) -> InnerState:
    """Resample the inner particles for a single trajectory.

    Args:
        rng_key: PRNGKey
        state: InnerState
            The state associated with a single outer trajectory.
            Leaves have shape (M, ...).
    """
    num_particles = state.particles.shape[0]
    resampling_idx = random.choice(
        rng_key, num_particles, shape=(num_particles,), p=state.weights
    )
    inner_state = InnerState(
        particles=state.particles[resampling_idx],
        log_weights=jnp.zeros(num_particles),
        weights=jnp.ones(num_particles) / num_particles,
        resampling_indices=resampling_idx,
    )
    return inner_state


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


def log_potential(
    reward_fn: RewardFn,
    particle: tuple[Array, Array],
    inner_state: InnerState,
    tempering: float,
) -> Array:
    r"""Estimate the log potential function.

    The estimate for the log potential function is given by
    .. math::
    \log g_t^n = \eta * \sum_{m=1}^M W_{s,t}^{nm} r_t(s_t^{nm}, a_t^n).

    Args:
        reward_fn: RewardFn
        particle: tuple[Array, Array]
            One outer particle $(z_t^n, a_t^n)$.
        inner_state: InnerState
            The inner state associated with the n-th outer trajectory.
            Leaves have shape (M, ...).
        tempering: float
            The tempering parameter
    """
    rewards = jax.vmap(reward_fn, in_axes=(0, None))(inner_state.particles, particle[1])
    return tempering * jnp.sum(rewards * inner_state.weights)


def init(
    rng_key: PRNGKey,
    inner_particles: Array,
    obs_model: ObservationModel,
    init_actions: Array,
    init_carry: list[LSTMCarry],
) -> tuple[OuterState, InnerState]:
    r"""Initialize the outer and inner states for the nested SMC algorithm.

    This samples from
    .. math::
      \begin{align}
        z_0^n &\sim \sum_{m=1}^M W_{s,0}^{nm} h(z_0 \mid s_0^{nm}), \\
      \end{align}

    for all $n \in \{1, \dots, N\}$.

    Args:
        rng_key: PRNGKey
        inner_particles: Array
            The initial state particles $s_0^{nm}$. Has shape (N, M, ...).
        obs_model: ObservationModel,
        init_actions: Array
            Init action particles $a_0^n$. Has shape (N, ...).
        init_carry: list[LSTMCarry]
            Init list of carry of the recurrent policy.
    """
    num_outer_particles, num_inner_particles = inner_particles.shape[:2]
    inner_state = InnerState(
        particles=inner_particles,
        log_weights=jnp.zeros(inner_particles.shape[:2]),
        weights=jnp.ones(inner_particles.shape[:2]) / num_inner_particles,
        resampling_indices=jnp.zeros(inner_particles.shape[:2], dtype=jnp.int32),
    )

    keys = random.split(rng_key, num_outer_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )

    outer_particles = (observations, init_actions, init_carry)
    outer_state = OuterState(
        particles=outer_particles,
        weights=jnp.ones(num_outer_particles) / num_outer_particles,
        resampling_indices=jnp.zeros(num_outer_particles, dtype=jnp.int32),
    )

    return outer_state, inner_state


def step(
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    outer_state: OuterState,
    inner_state: InnerState,
    tempering: float,
) -> tuple[OuterState, InnerState]:
    r"""A single step of the nested SMC algorithm.

    Args:
        rng_key: PRNGKey
        trans_model: TransitionModel
            The transition model for the state, $f(s_t \mid s_{t-1}, a_{t-1})$.
        obs_model: ObservationModel
            The observation model, $g(z_t \mid s_t)$.
        policy: RecurrentPolicy
            The stochastic policy, $\pi_\phi$.
        params: Dict
            Parameters of recurrent policy $\phi$.
        reward_fn: RewardFn
            The reward function, $r(s_t, a_t)$.
        outer_state: OuterState
            Leaves have shape (N, ...).
        inner_state: InnerState
            Leaves have shape (N, M, ...).
        tempering: float
            The tempering parameter, $\eta$.
    """
    num_particles = outer_state.weights.shape[0]

    # 0. Sample action from policy.
    key, sub_key = random.split(rng_key)
    carry, actions = policy.sample(
        sub_key, outer_state.particles[0], outer_state.particles[2], params
    )

    # 1. Resample the outer particles.
    key, sub_key = random.split(key)
    resampling_idx = random.choice(
        sub_key, num_particles, shape=(num_particles,), p=outer_state.weights
    )
    particles = jax.tree.map(lambda x: x[resampling_idx], outer_state.particles)
    inner_state = jax.tree.map(lambda x: x[resampling_idx], inner_state)

    # 2. Resample the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_state = jax.vmap(resample_inner)(keys[1:], inner_state)

    # 3. Propagate the inner particles.
    keys = random.split(keys[0], num_particles + 1)
    inner_particles = jax.vmap(propagate_inner, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, inner_state.particles, particles[1]
    )
    inner_state = inner_state._replace(particles=inner_particles)

    # 4. Propagate the outer particles.
    keys = random.split(keys[0], num_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )
    particles = (observations, actions, carry)

    # 5. Reweight the inner particles.
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, particles[0]
    )

    # 6. Reweight the outer particles.
    log_weights = jax.vmap(log_potential, in_axes=(None, 0, 0, None))(
        reward_fn, (particles[0], particles[1]), inner_state, tempering
    )
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    outer_state = OuterState(
        particles=particles,
        weights=weights,
        resampling_indices=resampling_idx,
    )

    return outer_state, inner_state


def genealogy_tracing(
    rng_key: Array,
    outer_state: OuterState,
):
    num_steps, num_particles = outer_state.weights.shape

    # starting at the last time step
    last_state = jax.tree.map(lambda x: x[-1], outer_state.particles)

    def tracing_fn(carry, args):
        idx = carry
        particles, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = jax.tree.map(lambda x: x[a], particles)
        return a, ancestors

    _, states = jax.lax.scan(
        tracing_fn,
        jnp.arange(num_particles),
        (
            jax.tree.map(lambda x: x[:-1], outer_state.particles),
            outer_state.resampling_indices
        ),
        reverse=True
    )

    states = jax.tree.map(lambda x, y: jnp.vstack((x, y)), states, last_state)
    return states
