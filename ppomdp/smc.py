from typing import Callable, Dict

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Distribution
from jax import Array, random

from ppomdp.core import (
    InnerState,
    ObservationModel,
    OuterParticles,
    OuterState,
    RecurrentPolicy,
    RewardFn,
    TransitionModel,
)
from ppomdp.utils import ess, systematic_resampling


def resample_inner(rng_key: PRNGKey, inner_state: InnerState) -> InnerState:
    """Resample the inner particles for a single outer trajectory.

    Only resamples if the effective sample size is below 75% of the number of particles.

    Args:
        rng_key: The random number generator key.
        inner_state: The state associated with a single outer trajectory.
            Leaves have shape (M, ...).
    """
    num_particles = inner_state.particles.shape[0]

    def true_fn(state: InnerState) -> InnerState:
        resampling_idx = systematic_resampling(rng_key, state.weights, num_particles)
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
        ess(inner_state.weights) < 0.75 * num_particles, true_fn, false_fn, inner_state
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
    action: Array,
    inner_state: InnerState,
) -> Array:
    """
    Calculate the expected reward for a given particle and inner state.

    Args:
        reward_fn: RewardFn
            The reward function, r(s_{t], a_{t-1}).
        action: Array
            Action particle.
        inner_state: InnerState
            The inner state containing particles and weights.

    Returns:
        Array: The cumulative return for the given particle and inner state.
    """
    rewards = jax.vmap(reward_fn, in_axes=(0, None))(inner_state.particles, action)
    return jnp.sum(rewards * inner_state.weights)


def log_potential(
    reward_fn: RewardFn,
    action: Array,
    inner_state: InnerState,
    tempering: float,
) -> tuple[Array, Array]:
    r"""Estimate the log potential function.

    The estimate for the log potential function is given by
    .. math::
    \log g_t^n = \eta * \sum_{m=1}^M W_{s,t}^{nm} r_t(s_t^{nm}, a_t^n).

    Args:
        reward_fn: RewardFn
        action: Array
            Action particle $a_t^n$.
        inner_state: InnerState
            The inner state associated with the n-th outer trajectory.
            Leaves have shape (M, ...).
        tempering: float
            The tempering parameter
    """
    rewards = expected_reward(reward_fn, action, inner_state)
    return tempering * rewards, rewards


def resample_outer(
    rng_key: PRNGKey, outer_state: OuterState, resample: bool
) -> OuterState:
    num_particles = outer_state.weights.shape[0]

    def true_fn(state: OuterState) -> OuterState:
        resampling_idx = systematic_resampling(rng_key, state.weights, num_particles)
        resampled_particles = jax.tree.map(lambda x: x[resampling_idx], state.particles)
        resampled_rewards = state.rewards[resampling_idx]
        return OuterState(
            particles=resampled_particles,
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            rewards=resampled_rewards,
            resampling_indices=resampling_idx,
        )

    def false_fn(state: OuterState) -> OuterState:
        resampling_idx = jnp.arange(num_particles)
        return state._replace(resampling_indices=resampling_idx)

    predicate = resample and ess(outer_state.weights) < 0.75 * num_particles
    resampled_state = jax.lax.cond(predicate, true_fn, false_fn, outer_state)
    return resampled_state


def smc_init(
    rng_key: PRNGKey,
    num_outer_particles: int,
    num_inner_particles: int,
    prior_dist: Distribution,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
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
        params: Dict
            Parameters of the recurrent policy.
    """
    key, sub_key = random.split(rng_key)
    inner_particles = prior_dist.sample(
        seed=sub_key,
        sample_shape=(
            num_outer_particles,
            num_inner_particles,
        ),
    )

    inner_state = InnerState(
        particles=inner_particles,
        log_weights=jnp.zeros((num_outer_particles, num_inner_particles)),
        weights=jnp.ones((num_outer_particles, num_inner_particles))
        / num_inner_particles,
        resampling_indices=jnp.zeros(
            (num_outer_particles, num_inner_particles), dtype=jnp.int_
        ),
    )

    # sample marginal observations
    keys = random.split(key, num_outer_particles + 1)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys[1:], obs_model, inner_state
    )

    # reweight inner particles
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, observations
    )

    # sample actions from policy
    key, sub_key = random.split(keys[0])
    init_carry = policy.reset(num_outer_particles)
    carry, actions, log_probs = policy.sample_and_log_prob(
        sub_key, observations, init_carry, params
    )

    outer_particles = OuterParticles(observations, actions, carry, log_probs)
    outer_state = OuterState(
        particles=outer_particles,
        log_weights=jnp.zeros(num_outer_particles),
        weights=jnp.ones(num_outer_particles) / num_outer_particles,
        resampling_indices=jnp.arange(num_outer_particles),
        rewards=jnp.zeros(num_outer_particles),
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
    resample_fn: Callable,
    outer_state: OuterState,
    inner_state: InnerState,
) -> tuple[OuterState, InnerState, Array]:
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
        resample_fn: Callable
            The resampling function.
        outer_state: OuterState
            Leaves have shape (N, ...).
        inner_state: InnerState
            Leaves have shape (N, M, ...).
    """
    num_particles = outer_state.weights.shape[0]

    # 1. Resample the outer particles.
    key, sub_key = random.split(rng_key)
    outer_state = resample_outer(sub_key, outer_state, resample)
    particles = outer_state.particles
    resampling_idx = outer_state.resampling_indices
    inner_state = jax.tree.map(lambda x: x[resampling_idx], inner_state)

    # 2. Resample the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_state = jax.vmap(resample_inner)(keys[1:], inner_state)

    # 3. Propagate the inner particles.
    keys = random.split(keys[0], num_particles + 1)
    inner_particles = jax.vmap(propagate_inner, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, inner_state.particles, particles.actions
    )
    inner_state = inner_state._replace(particles=inner_particles)

    # 4. Propagate the outer particles.
    keys = random.split(keys[0], num_particles + 1)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys[1:], obs_model, inner_state
    )
    carry, actions, log_probs = policy.sample_and_log_prob(
        keys[0], observations, particles.carry, params
    )

    # 5. Reweight the inner particles.
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, observations
    )

    # 6. Reweight the outer particles.
    log_potentials, rewards = jax.vmap(log_potential, in_axes=(None, 0, 0, None))(
        reward_fn, particles.actions, inner_state, tempering
    )
    log_weights = log_potentials + outer_state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)

    # 7. Compute the normalizing constant increment.
    # Eq. 10.3 in Chopin and Papaspiliopoulos (2020).
    log_marginal = logsum_weights - jax.nn.logsumexp(outer_state.log_weights)

    outer_particles = OuterParticles(observations, actions, carry, log_probs)
    outer_state = OuterState(
        particles=outer_particles,
        log_weights=log_weights,
        weights=weights,
        rewards=rewards,
        resampling_indices=resampling_idx,
    )
    return outer_state, inner_state, log_marginal


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
    resample_fn: Callable = systematic_resampling,
) -> tuple[OuterState, InnerState, Array]:
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
        resample_fn: Callable
            The resampling function.

    Returns:
        tuple[OuterState, InnerState, Array]
            All outer and inner states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def smc_loop(carry: tuple[OuterState, InnerState, Array], rng_key: PRNGKey):
        outer_state, inner_state, log_marginal = carry

        outer_state, inner_state, log_marginal_incr = smc_step(
            rng_key,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            resample,
            resample_fn,
            outer_state,
            inner_state,
        )

        log_marginal += log_marginal_incr
        return (outer_state, inner_state, log_marginal), (outer_state, inner_state)

    init_key, loop_key = random.split(rng_key, 2)
    init_outer_state, init_inner_state = smc_init(
        init_key,
        num_outer_particles,
        num_inner_particles,
        prior_dist,
        obs_model,
        policy,
        params,
    )

    keys = random.split(loop_key, num_time_steps)
    (_, _, log_marginal), (outer_states, inner_states) = jax.lax.scan(
        smc_loop, (init_outer_state, init_inner_state, jnp.array(0.0)), keys
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    outer_states = concat_trees(init_outer_state, outer_states)
    inner_states = concat_trees(init_inner_state, inner_states)
    return outer_states, inner_states, log_marginal


def backward_tracing(
    rng_key: Array,
    outer_states: OuterState,
    inner_states: InnerState,
    sample: bool = True,
) -> tuple[OuterState, InnerState]:
    """Genealogy tracking to get the smoothed trajectories.

    Args:
        rng_key: The random number generator key.
        outer_states: The outer states from the output of the SMC algorithm.
        inner_states: The inner states from the output of the SMC algorithm.
        sample: If True, sample the genealogy, otherwise trace back all final
          particles.

    Returns:
        The traced outer and inner states.
    """
    num_steps, num_particles = outer_states.weights.shape

    # Sample the states at the final time step.
    resampling_idx = jax.lax.cond(
        sample,
        lambda _: random.choice(
            rng_key, num_particles, shape=(num_particles,), p=outer_states.weights[-1]
        ),
        lambda _: jnp.arange(num_particles),
        None,
    )

    last_outer = jax.tree.map(lambda x: x[-1, resampling_idx], outer_states)
    last_inner = jax.tree.map(lambda x: x[-1, resampling_idx], inner_states)

    # Trace the genealogy for the outer states.
    def tracing_fn(carry, args):
        idx = carry
        states, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = jax.tree.map(lambda x: x[a], states)
        return a, (a, ancestors)

    _, (traced_indices, traced_outer) = jax.lax.scan(
        tracing_fn,
        resampling_idx,
        (
            jax.tree.map(lambda x: x[:-1], outer_states),
            outer_states.resampling_indices[1:],
        ),
        reverse=True,
    )

    # Trace the inner states.
    def get_traced_inner(idx, state):
        return jax.tree.map(lambda x: x[idx], state)

    traced_inner = jax.vmap(get_traced_inner, in_axes=(0, 0))(
        traced_indices,
        jax.tree.map(lambda x: x[:-1], inner_states),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    traced_outer = concat_trees(traced_outer, last_outer)
    traced_inner = concat_trees(traced_inner, last_inner)
    return traced_outer, traced_inner
