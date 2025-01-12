from typing import Callable, Dict

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
    systematic_resampling,
    policy_logpdf,
    marginal_observation_logpdf,
    transition_logpdf,
)


def smc_init(
    rng_key: PRNGKey,
    num_outer_particles: int,
    num_inner_particles: int,
    prior_dist: Distribution,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
) -> tuple[OuterState, InnerState, InnerInfo]:
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
        sample_shape=(num_outer_particles, num_inner_particles),
    )

    inner_state = InnerState(
        particles=inner_particles,
        log_weights=jnp.zeros((num_outer_particles, num_inner_particles)),
        weights=jnp.ones((num_outer_particles, num_inner_particles)) / num_inner_particles,
        resampling_indices=jnp.zeros((num_outer_particles, num_inner_particles), dtype=jnp.int_),
    )

    # sample marginal observations
    keys = random.split(key, num_outer_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )

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
    dummy_log_probs = jnp.zeros(num_outer_particles)

    outer_particles = OuterParticles(observations, dummy_actions, init_carry, dummy_log_probs)
    outer_state = OuterState(
        particles=outer_particles,
        log_weights=jnp.zeros(num_outer_particles),
        weights=jnp.ones(num_outer_particles) / num_outer_particles,
        resampling_indices=jnp.zeros(num_outer_particles, dtype=jnp.int_),
        rewards=jnp.zeros(num_outer_particles),
    )

    return outer_state, inner_state, inner_info


def smc_step(
    time_idx: int,
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
    resample: bool,
    resample_fn: Callable,
    outer_state: OuterState,
    inner_state: InnerState,
) -> tuple[OuterState, InnerState, InnerInfo, Array]:
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
        slew_rate_penalty: float
            The slew rate penalty.
        resample: bool
            If True, resample, otherwise do not resample.
        resample_fn: Callable
            The resampling function.
        outer_state: OuterState
            Leaves have shape (N, ...).
        inner_state: InnerState
            Leaves have shape (N, M, ...).
        time_idx: int
            The current time index.
    """
    num_particles = outer_state.weights.shape[0]

    # 1. Resample the outer particles.
    key, sub_key = random.split(rng_key)
    outer_state = resample_outer(sub_key, outer_state, resample, resample_fn)
    particles = outer_state.particles
    resampling_idx = outer_state.resampling_indices
    inner_state = jax.tree.map(lambda x: x[resampling_idx], inner_state)

    # 2. Resample the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_state = jax.vmap(resample_inner, in_axes=(0, 0, None))(
        keys[1:], inner_state, resample_fn
    )

    # 3. Sample new actions.
    key, sub_key = random.split(keys[0])
    carry, actions, log_probs = policy.sample_and_log_prob(
        sub_key, particles.observations, particles.carry, params
    )

    # 4. Propagate the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_particles = jax.vmap(propagate_inner, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, inner_state.particles, actions
    )
    inner_state = inner_state._replace(particles=inner_particles)

    # 5. Sample new observations.
    keys = random.split(keys[0], num_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, inner_state
    )

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


def smc(
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
    resample: bool = True,
    resample_fn: Callable = systematic_resampling
) -> tuple[OuterState, InnerState, InnerInfo, Array]:
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
        slew_rate_penalty: float
            The slew rate penalty.
        resample: bool
            If True, resample, otherwise do not resample.
        resample_fn: Callable
            The resampling function.

    Returns:
        tuple[OuterState, InnerState, Array]
            All outer and inner states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def smc_loop(carry: tuple[OuterState, InnerState, Array], args: tuple[int, PRNGKey]):
        outer_state, inner_state, log_marginal = carry
        time_idx, key = args

        outer_state, inner_state, inner_info, log_marginal_incr = smc_step(
            time_idx,
            key,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty,
            resample,
            resample_fn,
            outer_state,
            inner_state,
        )

        log_marginal += log_marginal_incr
        return (outer_state, inner_state, log_marginal), (outer_state, inner_state, inner_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_outer_state, init_inner_state, init_inner_info = smc_init(
        init_key,
        num_outer_particles,
        num_inner_particles,
        prior_dist,
        obs_model,
        policy,
        params,
    )

    time_indices = jnp.arange(num_time_steps)
    keys = random.split(loop_key, num_time_steps)
    (_, _, log_marginal), (outer_states, inner_states, inner_infos) = jax.lax.scan(
        smc_loop,
        (init_outer_state, init_inner_state, jnp.array(0.0)),
        (time_indices, keys),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    outer_states = concat_trees(init_outer_state, outer_states)
    inner_states = concat_trees(init_inner_state, inner_states)
    inner_infos = concat_trees(init_inner_info, inner_infos)
    return outer_states, inner_states, inner_infos, log_marginal


def backward_tracing(
    rng_key: Array,
    outer_states: OuterState,
    inner_states: InnerState,
    inner_infos: InnerInfo,
    resample: bool = True,
    resample_fn: Callable = systematic_resampling,
) -> tuple[OuterState, InnerState, InnerInfo]:
    """Genealogy tracking to get the smoothed trajectories.

    Args:
        rng_key: The random number generator key.
        outer_states: The outer states of the outer particle filter.
        inner_states: The inner states of the inner particle filter.
        inner_infos: The inner infos of the inner particle filter.
        resample: If True, sample the genealogy, otherwise trace back all final
          particles.
        resample_fn: The resampling function.
    Returns:
        The traced outer and inner states.
    """
    num_steps, num_particles = outer_states.weights.shape

    # Sample the states at the final time step.
    resampling_idx = jax.lax.cond(
        resample,
        lambda _: resample_fn(rng_key, outer_states.weights[-1], num_particles),
        lambda _: jnp.arange(num_particles),
        None,
    )

    last_outer_state = jax.tree.map(lambda x: x[-1, resampling_idx], outer_states)
    last_inner_state = jax.tree.map(lambda x: x[-1, resampling_idx], inner_states)
    last_inner_info = jax.tree.map(lambda x: x[-1, resampling_idx], inner_infos)

    # Trace the genealogy for the outer states.
    def tracing_fn(carry, args):
        idx = carry
        states, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = jax.tree.map(lambda x: x[a], states)
        return a, (a, ancestors)

    _, (traced_indices, traced_outer_states) = jax.lax.scan(
        tracing_fn,
        resampling_idx,
        (
            jax.tree.map(lambda x: x[:-1], outer_states),
            outer_states.resampling_indices[1:],
        ),
        reverse=True,
    )

    # Trace the inner states.^
    def get_traced_inner(idx, state, info):
        return jax.tree.map(lambda x: x[idx], state), jax.tree.map(lambda x: x[idx], info)

    traced_inner_states, traced_inner_infos = jax.vmap(get_traced_inner, in_axes=(0, 0, 0))(
        traced_indices,
        jax.tree.map(lambda x: x[:-1], inner_states),
        jax.tree.map(lambda x: x[:-1], inner_infos),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    traced_outer_states = concat_trees(traced_outer_states, last_outer_state)
    traced_inner_states = concat_trees(traced_inner_states, last_inner_state)
    traced_inner_infos = concat_trees(traced_inner_infos, last_inner_info)
    return traced_outer_states, traced_inner_states, traced_inner_infos


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


def mcmc_backward_sampling_single(
    rng_key: PRNGKey,
    outer_state: OuterState,
    inner_state: InnerState,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
):
    """
    MCMC-based backward sampling from Bunch and Godsill (2013)
    """
    num_time_steps, num_outer_particles = outer_state.weights.shape

    # Sample the last particle
    key, sub_key = random.split(rng_key)
    idx = jax.random.choice(
        sub_key, jnp.arange(num_outer_particles), p=outer_state.weights[-1]
    )
    last_sampled_outer_state = jax.tree.map(lambda x: x[-1, idx], outer_state)
    last_sampled_inner_state = jax.tree.map(lambda x: x[-1, idx], inner_state)

    def body(idx, args):
        t, key = args

        ancestor_idx = outer_state.resampling_indices[t + 1, idx]
        key, sub_key = random.split(key)
        proposed_idx = jax.random.choice(
            sub_key, jnp.arange(num_outer_particles), p=outer_state.weights[t]
        )

        ancestor_policy_logpdfs = policy_logpdf(
            stitch_arrays(
                outer_state.particles.actions[:, ancestor_idx],
                outer_state.particles.actions[:, idx],
                stitch=t, max_size=num_time_steps
            ),
            stitch_arrays(
                outer_state.particles.observations[:, ancestor_idx],
                outer_state.particles.observations[:, idx],
                stitch=t, max_size=num_time_steps
            ),
            policy, params
        )
        ancestor_policy_logpdf = jnp.sum(
            jnp.where(
                jnp.arange(num_time_steps) < t,
                jnp.zeros((num_time_steps,)),
                ancestor_policy_logpdfs
            )
        )

        proposed_policy_logpdfs = policy_logpdf(
            stitch_arrays(
                outer_state.particles.actions[:, proposed_idx],
                outer_state.particles.actions[:, idx],
                stitch=t + 1, max_size=num_time_steps
            ),
            stitch_arrays(
                outer_state.particles.observations[:, proposed_idx],
                outer_state.particles.observations[:, idx],
                stitch=t + 1, max_size=num_time_steps
            ),
            policy, params
        )
        proposed_policy_logpdf = jnp.sum(
            jnp.where(
                jnp.arange(num_time_steps) < t,
                jnp.zeros((num_time_steps,)),
                proposed_policy_logpdfs
            )
        )

        ancestor_trans_logpdf = transition_logpdf(
            inner_state.particles[t, ancestor_idx],
            inner_state.log_weights[t, ancestor_idx],
            outer_state.particles.actions[t + 1, idx],
            inner_state.particles[t + 1, idx],
            trans_model,
            obs_model
        )
        proposed_trans_logpdf = transition_logpdf(
            inner_state.particles[t, proposed_idx],
            inner_state.log_weights[t, proposed_idx],
            outer_state.particles.actions[t + 1, idx],
            inner_state.particles[t + 1, idx],
            trans_model,
            obs_model
        )

        ancestor_log_potential, _ = log_potential(
            reward_fn,
            jax.tree.map(lambda x: x[t + 1, idx], inner_state),
            outer_state.particles.actions[t + 1, idx],
            outer_state.particles.actions[t, ancestor_idx],
            t + 1,
            tempering,
            slew_rate_penalty
        )
        proposed_log_potential, _ = log_potential(
            reward_fn,
            jax.tree.map(lambda x: x[t + 1, idx], inner_state),
            outer_state.particles.actions[t + 1, idx],
            outer_state.particles.actions[t, proposed_idx],
            t + 1,
            tempering,
            slew_rate_penalty
        )

        log_prob_ancestor = (
            ancestor_policy_logpdf + ancestor_trans_logpdf + ancestor_log_potential
        )
        log_prob_proposed = (
            proposed_policy_logpdf + proposed_trans_logpdf + proposed_log_potential
        )

        # accept / reject
        key, sub_key = random.split(key)
        log_u = jnp.log(jax.random.uniform(sub_key))
        log_ratio = log_prob_proposed - log_prob_ancestor
        idx = jax.lax.select(log_ratio > log_u, proposed_idx, ancestor_idx)
        return idx, (
            jax.tree.map(lambda x: x[t, idx], outer_state),
            jax.tree.map(lambda x: x[t, idx], inner_state)
        )

    _, (sampled_outer_states, sampled_inner_states) = jax.lax.scan(
        body, idx,
        (
            jnp.arange(num_time_steps - 1),
            random.split(key, num_time_steps - 1)
        ), reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    sampled_outer_states = concat_trees(sampled_outer_states, last_sampled_outer_state)
    sampled_inner_states = concat_trees(sampled_inner_states, last_sampled_inner_state)
    return sampled_outer_states, sampled_inner_states


def mcmc_backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    outer_state: OuterState,
    inner_state: InnerState,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Dict,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
):
    keys = random.split(rng_key, num_samples)
    sampled_outer_states, sampled_inner_states = jax.vmap(
        lambda key: mcmc_backward_sampling_single(
            key,
            outer_state,
            inner_state,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty),
        out_axes=1)(keys)
    return sampled_outer_states, sampled_inner_states
