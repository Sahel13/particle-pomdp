from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array, random

from chex import PRNGKey
from distrax import Distribution

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


def expected_reward(
    reward_fn: RewardFn,
    particle: tuple[Array, Array],
    inner_state: InnerState,
) -> Array:
    """
    Calculate the expected reward for a given particle and inner state.

    Args:
        reward_fn: RewardFn
            The reward function, r(s_{t], a_{t-1}).
        particle: tuple[Array, Array]
            A tuple containing the observation and action for the particle.
        inner_state: InnerState
            The inner state containing particles and weights.

    Returns:
        Array: The cumulative return for the given particle and inner state.
    """
    rewards = jax.vmap(reward_fn, in_axes=(0, None))(inner_state.particles, particle[1])
    return jnp.sum(rewards * inner_state.weights)


def log_potential(
    reward_fn: RewardFn,
    particle: tuple[Array, Array],
    inner_state: InnerState,
    tempering: float,
) -> tuple[Array, Array]:
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
    rewards = expected_reward(reward_fn, particle, inner_state)
    return tempering * rewards, rewards


def smc_init(
    rng_key: PRNGKey,
    num_outer_particles: int,
    num_inner_particles: int,
    prior_dist: Distribution,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
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
        num_outer_particles: int
            The number of outer particles $N$.
        num_inner_particles: int
            The number of inner particles $M$.
        prior_dist: distrax.Distribution
            The prior distribution for the initial state particles.
        obs_model: ObservationModel
            The observation model.
        policy: RecurrentPolicy
            The recurrent policy.
    """
    key, sub_key = random.split(rng_key)
    inner_particles = prior_dist.sample(
        seed=sub_key,
        sample_shape=(num_outer_particles, num_inner_particles,)
    )

    inner_state = InnerState(
        particles=inner_particles,
        log_weights=jnp.zeros(inner_particles.shape[:2]),
        weights=jnp.ones((num_outer_particles, num_inner_particles)) / num_inner_particles,
        resampling_indices=jnp.zeros((num_outer_particles, num_inner_particles), dtype=jnp.int_),
    )

    keys = random.split(key, num_outer_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )

    init_carry = policy.reset(num_outer_particles)
    init_actions = jnp.zeros((num_outer_particles, policy.dim))
    outer_particles = (observations, init_actions, init_carry)
    outer_state = OuterState(
        particles=outer_particles,
        weights=jnp.ones(num_outer_particles) / num_outer_particles,
        rewards=jnp.zeros(num_outer_particles),
        resampling_indices=jnp.zeros(num_outer_particles, dtype=jnp.int_),
    )

    return outer_state, inner_state


def smc_step(
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    resample: bool,
    outer_state: OuterState,
    inner_state: InnerState,
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
        tempering: float
            The tempering parameter, $\eta$.
        resample: bool
            If True, resample, otherwise do not resample.
        outer_state: OuterState
            Leaves have shape (N, ...).
        inner_state: InnerState
            Leaves have shape (N, M, ...).
    """
    num_particles = outer_state.weights.shape[0]

    # 0. Sample action from policy.
    key, sub_key = random.split(rng_key)
    carry, actions = policy.sample(
        sub_key, outer_state.particles[0], outer_state.particles[2], params
    )
    particles = (outer_state.particles[0], actions, carry)  # (y_{t-1}, a_{t-1}, c_{t-1})

    # 1. Resample the outer particles.
    key, sub_key = random.split(key)
    resampling_idx = jax.lax.cond(
        resample,
        lambda _: random.choice(sub_key, num_particles, shape=(num_particles,), p=outer_state.weights),
        lambda _: jnp.arange(num_particles),
        operand=None
    )

    particles = jax.tree.map(lambda x: x[resampling_idx], particles)
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

    # 5. Reweight the inner particles.
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, observations
    )

    # 6. Reweight the outer particles.
    log_weights, rewards = jax.vmap(log_potential, in_axes=(None, 0, 0, None))(
        reward_fn, (observations, particles[1]), inner_state, tempering
    )
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    outer_state = OuterState(
        particles=(observations, particles[1], particles[2]),
        weights=weights,
        rewards=rewards,
        resampling_indices=resampling_idx,
    )

    return outer_state, inner_state


def smc(
    rng_key: PRNGKey,
    num_outer_particles: int,
    num_inner_particles: int,
    num_time_steps: int,
    prior_dist: Distribution,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    resample: bool = True,
) -> tuple[OuterState, InnerState]:
    """
    Perform the Sequential Monte Carlo (SMC) algorithm.

    Args:
        rng_key: PRNGKey
            The random key for sampling.
        num_outer_particles: int
            The number of outer particles.
        num_inner_particles: int
            The number of inner particles.
        num_time_steps: int
            The number of time steps for the SMC algorithm.
        prior_dist: Distribution
            The prior distribution for the initial state particles.
        trans_model: TransitionModel
            The transition model for the state.
        obs_model: ObservationModel
            The observation model.
        policy: RecurrentPolicy
            The recurrent policy.
        params: Dict
            Parameters of the recurrent policy.
        reward_fn: RewardFn
            The reward function.
        tempering: float
            The tempering parameter.
        resample: bool
            If True, resample, otherwise do not resample.

    Returns:
        tuple[OuterState, InnerState]
            The final outer and inner states after running the SMC algorithm.
    """
    def smc_loop(carry: tuple[OuterState, InnerState], rng_key: PRNGKey):
        outer_state, inner_state = carry

        outer_state, inner_state = smc_step(
            rng_key,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            resample,
            outer_state,
            inner_state,
        )

        return (outer_state, inner_state), (outer_state, inner_state)

    key, init_key, step_key = random.split(rng_key, 3)
    init_outer_state, init_inner_state = smc_init(
        init_key,
        num_outer_particles,
        num_inner_particles,
        prior_dist,
        obs_model,
        policy
    )
    _, (outer_states, inner_states) = jax.lax.scan(
        smc_loop,
        (init_outer_state, init_inner_state),
        random.split(step_key, num_time_steps)
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    outer_states = concat_trees(init_outer_state, outer_states)
    inner_states = concat_trees(init_inner_state, inner_states)
    return outer_states, inner_states


def backward_tracing(
    rng_key: Array,
    outer_state: OuterState,
) -> tuple[Array, Array, list[LSTMCarry]]:
    """
    Perform backward tracing to trace the ancestors of the outer states.

    Args:
        rng_key: Array
            The random key for sampling.
        outer_state: OuterState
            The outer state containing particles and weights.

    Returns:
        tuple[Array, Array, list[LSTMCarry]
            The traced observations, actions and carry.
    """
    num_steps, num_particles = outer_state.weights.shape

    # resample according to last weights
    resampling_idx = random.choice(
        rng_key, num_particles,
        shape=(num_particles,),
        p=outer_state.weights[-1]
    )
    last_state = jax.tree.map(lambda x: x[-1, resampling_idx], outer_state.particles)

    def tracing_fn(carry, args):
        idx = carry
        particles, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = jax.tree.map(lambda x: x[a], particles)
        return a, ancestors

    _, traced_states = jax.lax.scan(
        tracing_fn,
        jnp.arange(num_particles),
        (
            jax.tree.map(lambda x: x[:-1], outer_state.particles),
            outer_state.resampling_indices[1:]
        ),
        reverse=True
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    traced_states = concat_trees(traced_states, last_state)
    return traced_states
