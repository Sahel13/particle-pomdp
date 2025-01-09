from typing import Dict

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Distribution
from jax import Array, random

from ppomdp.core import (
    InnerState,
    InnerInfo,
    ObservationModel,
    OuterParticles,
    OuterState,
    Reference,
    RecurrentPolicy,
    RewardFn,
    TransitionModel,
)
from ppomdp.utils import (
    resample_inner,
    resample_outer,
    propagate_inner,
    reweight_inner,
    sample_marginal_obs,
    log_potential,
    weighted_mean,
    weighted_covar,
    effective_sample_size,
    multinomial_resampling,
    systematic_resampling
)


def csmc_init(
    rng_key: PRNGKey,
    num_outer_particles: int,
    num_inner_particles: int,
    prior_dist: Distribution,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reference: Reference,
) -> tuple[OuterState, InnerState, InnerInfo]:
    r"""Initialize the outer and inner states for the nested CSMC algorithm.

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
        reference: Reference
            Reference state of the conditional particle filter.
    """
    key, sub_key = random.split(rng_key)
    inner_particles = prior_dist.sample(
        seed=sub_key,
        sample_shape=(num_outer_particles, num_inner_particles),
    )

    inner_state = InnerState(
        particles=inner_particles,
        log_weights=jnp.zeros((num_outer_particles, num_inner_particles)),
        weights=jnp.ones((num_outer_particles, num_inner_particles)) / num_inner_particles,
        resampling_indices=jnp.zeros((num_outer_particles, num_inner_particles), dtype=jnp.int_),
    )

    # replace zeroth inner state with reference inner state
    inner_state = jax.tree.map(
        lambda x, y: x.at[0].set(y), inner_state, reference.inner_state
    )

    # sample marginal observations
    keys = random.split(key, num_outer_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )

    # replace zeroth observations with reference observation
    observations = observations.at[0].set(reference.observations)

    # reweight inner particles
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, observations
    )
    inner_info = InnerInfo(
        ess=effective_sample_size(inner_state.weights),
        mean=weighted_mean(inner_state.particles, inner_state.weights),
        covar=weighted_covar(inner_state.particles, inner_state.weights)
    )

    # Initialize dummy actions and policy carry.
    init_carry = policy.reset(num_outer_particles)
    dummy_actions = jnp.zeros((num_outer_particles, policy.dim))
    init_log_probs = jnp.zeros(num_outer_particles)

    # replace zeroth carry, action, and log_prob with reference carry, action, and log_prob
    init_carry = jax.tree_map(lambda x, y: x.at[0].set(y), init_carry, reference.carry)
    dummy_actions = dummy_actions.at[0].set(reference.actions)
    init_log_probs = init_log_probs.at[0].set(reference.log_probs)

    outer_particles = OuterParticles(observations, dummy_actions, init_carry, init_log_probs)
    outer_state = OuterState(
        particles=outer_particles,
        log_weights=jnp.zeros(num_outer_particles),
        weights=jnp.ones(num_outer_particles) / num_outer_particles,
        resampling_indices=jnp.zeros(num_outer_particles, dtype=jnp.int_),
        rewards=jnp.zeros(num_outer_particles),
    )
    return outer_state, inner_state, inner_info


def csmc_step(
    time_idx: int,
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
    reference: Reference,
    resample: bool,
    outer_state: OuterState,
    inner_state: InnerState,
) -> tuple[OuterState, InnerState, InnerInfo, Array]:
    r"""A single step of the nested CSMC algorithm.

    Args:
        time_idx: int
            The current time index.
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
        slew_rate_penalty: float
            The slew rate penalty.
        reference: Reference
            Reference trajectory of the conditional particle filter.
        resample: bool
            If True, resample, otherwise do not resample.
            The resampling function.
        outer_state: OuterState
            Leaves have shape (N, ...).
        inner_state: InnerState
            Leaves have shape (N, M, ...).
    """
    num_particles = outer_state.weights.shape[0]

    # 1. Resample the outer particles.
    key, sub_key = random.split(rng_key)
    outer_state = resample_outer(
        sub_key, outer_state, resample, multinomial_resampling, conditional=True
    )
    particles = outer_state.particles
    resampling_idx = outer_state.resampling_indices
    inner_state = jax.tree.map(lambda x: x[resampling_idx], inner_state)

    # 2. Resample the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_state = jax.vmap(resample_inner, in_axes=(0, 0, None))(
        keys[1:], inner_state, systematic_resampling
    )

    # 3. Sample new actions.
    key, sub_key = random.split(keys[0])
    carry, actions, log_probs = policy.sample_and_log_prob(
        sub_key, particles.observations, particles.carry, params
    )

    # replace zeroth carry, action, and log_prob with reference carry, action, and log_prob
    carry = jax.tree.map(lambda x, y: x.at[0].set(y), carry, reference.carry)
    actions = actions.at[0].set(reference.actions)
    log_probs = log_probs.at[0].set(reference.log_probs)

    # 4. Propagate the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_particles = jax.vmap(propagate_inner, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, inner_state.particles, actions
    )
    inner_state = inner_state._replace(particles=inner_particles)

    # replace zeroth inner state with reference inner state
    inner_state = jax.tree.map(
        lambda x, y: x.at[0].set(y), inner_state, reference.inner_state
    )

    # 5. Sample new observations.
    keys = random.split(keys[0], num_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )

    # replace zeroth observations with reference observation
    observations = observations.at[0].set(reference.observations)

    # 6. Reweight the inner particles.
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, observations
    )
    inner_info = InnerInfo(
        ess=effective_sample_size(inner_state.weights),
        mean=weighted_mean(inner_state.particles, inner_state.weights),
        covar=weighted_covar(inner_state.particles, inner_state.weights)
    )

    # 7. Reweight the outer particles.
    log_potentials, rewards = jax.vmap(log_potential, in_axes=(None, 0, 0, 0, None, None, None))(
        reward_fn, inner_state, actions, particles.actions, time_idx, tempering, slew_rate_penalty
    )
    log_weights = log_potentials + outer_state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)

    # 8. Compute the normalizing constant increment.
    # Eq. 10.3 in Chopin and Papaspiliopoulos (2020).
    log_marginal = logsum_weights - jax.nn.logsumexp(outer_state.log_weights)

    outer_particles = OuterParticles(observations, actions, carry, log_probs)
    outer_state = OuterState(
        particles=outer_particles,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_idx,
        rewards=rewards,
    )
    return outer_state, inner_state, inner_info, log_marginal


def csmc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_outer_particles: int,
    num_inner_particles: int,
    prior_dist: Distribution,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
    reference: Reference,
    resample: bool = True,
) -> tuple[OuterState, InnerState, InnerInfo, Array]:
    """
    Perform the Conditional Sequential Monte Carlo (CSMC) algorithm.

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
        slew_rate_penalty: float
            The slew rate penalty.
        reference: Reference
            Reference trajectory of the conditional particle filter.
        resample: bool
            If True, resample, otherwise do not resample.
            The resampling function.

    Returns:
        tuple[OuterState, InnerState, Array]
            All outer and inner states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def csmc_loop(carry, args):
        outer_state, inner_state, log_marginal = carry
        time_idx, key, ref_step = args

        outer_state, inner_state, inner_info, log_marginal_incr = csmc_step(
            time_idx,
            key,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty,
            ref_step,
            resample,
            outer_state,
            inner_state,
        )

        log_marginal += log_marginal_incr
        return (outer_state, inner_state, log_marginal), (outer_state, inner_state, inner_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_outer_state, init_inner_state, init_inner_info = csmc_init(
        init_key,
        num_outer_particles,
        num_inner_particles,
        prior_dist,
        obs_model,
        policy,
        params,
        jax.tree.map(lambda x: x[0], reference),
    )

    (_, _, log_marginal), (outer_states, inner_states, inner_infos) = jax.lax.scan(
        csmc_loop,
        (init_outer_state, init_inner_state, jnp.array(0.0)),
        (
            jnp.arange(num_time_steps),  # time indices
            random.split(loop_key, num_time_steps),  # random keys
            jax.tree.map(lambda x: x[1:], reference)  # reference trajectory
        ),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    outer_states = concat_trees(init_outer_state, outer_states)
    inner_states = concat_trees(init_inner_state, inner_states)
    inner_infos = concat_trees(init_inner_info, inner_infos)
    return outer_states, inner_states, inner_infos, log_marginal
