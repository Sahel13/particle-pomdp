from functools import partial
from typing import Callable, Dict

import jax
from jax import Array, random, numpy as jnp

from distrax import Distribution

from ppomdp.core import (
    BeliefState,
    BeliefInfo,
    ObservationModel,
    HistoryParticles,
    HistoryState,
    RecurrentPolicy,
    RewardFn,
    TransitionModel,
    Parameters,
    PRNGKey
)
from ppomdp.utils import (
    resample_belief,
    resample_history,
    propagate_belief,
    reweight_belief,
    sample_marginal_obs,
    log_potential,
    weighted_mean,
    weighted_covar,
    effective_sample_size,
    systematic_resampling,
    policy_logpdf,
    transition_logpdf,
)


def smc_init(
    rng_key: PRNGKey,
    num_history_particles: int,
    num_belief_particles: int,
    prior_dist: Distribution,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Parameters,
) -> tuple[HistoryState, BeliefState, BeliefInfo]:
    r"""Initialize the history and belief states for the nested SMC algorithm.

    This samples from
    .. math::
      \begin{align}
        z_0^n &\sim \sum_{m=1}^M W_{s,0}^{nm} h(z_0 \mid s_0^{nm}), \\
      \end{align}

    for all $n \in \{1, \dots, N\}$.

    Args:
        rng_key: PRNGKey
        num_history_particles: int
            The number of history particles $N$.
        num_belief_particles: int
            The number of belief particles $M$.
        prior_dist: distrax.Distribution
            The prior distribution for the initial state particles.
        obs_model: ObservationModel
            The observation model.
        policy: RecurrentPolicy
            The recurrent policy.
        params: Parameters
            Parameters of the recurrent policy.
    """
    key, sub_key = random.split(rng_key)
    belief_particles = prior_dist.sample(
        seed=sub_key,
        sample_shape=(num_history_particles, num_belief_particles),
    )

    belief_state = BeliefState(
        particles=belief_particles,
        log_weights=jnp.zeros((num_history_particles, num_belief_particles)),
        weights=jnp.ones((num_history_particles, num_belief_particles)) / num_belief_particles,
        resampling_indices=jnp.zeros((num_history_particles, num_belief_particles), dtype=jnp.int32),
    )

    # sample marginal observations
    keys = random.split(key, num_history_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, belief_state
    )

    # reweight belief particles
    belief_state = jax.vmap(reweight_belief, in_axes=(None, 0, 0))(
        obs_model, belief_state, observations
    )
    belief_info = BeliefInfo(
        ess=effective_sample_size(belief_state.log_weights),
        mean=weighted_mean(belief_state.particles, belief_state.weights),
        covar=weighted_covar(belief_state.particles, belief_state.weights)
    )

    # Initialize dummy actions and policy carry.
    init_carry = policy.reset(num_history_particles)
    dummy_actions = jnp.zeros((num_history_particles, policy.dim))
    dummy_log_probs = jnp.zeros(num_history_particles)

    history_particles = HistoryParticles(observations, dummy_actions, init_carry, dummy_log_probs)
    history_state = HistoryState(
        particles=history_particles,
        log_weights=jnp.zeros(num_history_particles),
        weights=jnp.ones(num_history_particles) / num_history_particles,
        resampling_indices=jnp.zeros(num_history_particles, dtype=jnp.int32),
        rewards=jnp.zeros(num_history_particles),
    )

    return history_state, belief_state, belief_info


def smc_step(
    time_idx: int,
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
    resample_fn: Callable,
    history_state: HistoryState,
    belief_state: BeliefState,
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    r"""A single step of the nested SMC algorithm.

    Args:
        rng_key: PRNGKey
        trans_model: TransitionModel
            The transition model for the state, $f(s_t \mid s_{t-1}, a_{t-1})$.
        obs_model: ObservationModel
            The observation model, $g(z_t \mid s_t)$.
        policy: RecurrentPolicy
            The stochastic policy, $\pi_\phi$.
        params: Parameters
            Parameters of recurrent policy $\phi$.
        reward_fn: RewardFn
            The reward function, $r(s_t, a_t)$.
        tempering: float
            The tempering parameter, $\eta$.
        slew_rate_penalty: float
            The slew rate penalty.
        resample_fn: Callable
            The resampling function.
        history_state: HistoryState
            Leaves have shape (N, ...).
        belief_state: BeliefState
            Leaves have shape (N, M, ...).
        time_idx: int
            The current time index.
    """
    num_particles = history_state.weights.shape[0]

    # 1. Resample the history particles.
    key, sub_key = random.split(rng_key)
    history_state = resample_history(sub_key, history_state, resample_fn)
    particles = history_state.particles
    resampling_idx = history_state.resampling_indices
    belief_state = jax.tree.map(lambda x: x[resampling_idx], belief_state)

    # 2. Resample the belief particles.
    keys = random.split(key, num_particles + 1)
    belief_state = jax.vmap(resample_belief, in_axes=(0, 0, None))(
        keys[1:], belief_state, resample_fn
    )

    # 3. Sample new actions.
    key, sub_key = random.split(keys[0])
    carry, actions, log_probs = policy.sample_and_log_prob(
        sub_key, particles.carry, particles.observations, params
    )

    # 4. Propagate the belief particles.
    keys = random.split(key, num_particles + 1)
    belief_particles = jax.vmap(propagate_belief, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, belief_state.particles, actions
    )
    belief_state = belief_state._replace(particles=belief_particles)

    # 5. Sample new observations.
    keys = random.split(keys[0], num_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, belief_state
    )

    # 6. Reweight the belief particles.
    belief_state = jax.vmap(reweight_belief, in_axes=(None, 0, 0))(
        obs_model, belief_state, observations
    )
    belief_info = BeliefInfo(
        ess=effective_sample_size(belief_state.log_weights),
        mean=weighted_mean(belief_state.particles, belief_state.weights),
        covar=weighted_covar(belief_state.particles, belief_state.weights)
    )

    # 7. Reweight the history particles.
    log_potentials, rewards = jax.vmap(log_potential, in_axes=(0, 0, 0, None, None, None, None))(
        belief_state, actions, particles.actions, time_idx, reward_fn, tempering, slew_rate_penalty
    )
    log_weights = log_potentials + history_state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)

    # 8. Compute the normalizing constant increment.
    # Eq. 10.3 in Chopin and Papaspiliopoulos (2020).
    log_marginal = logsum_weights - jax.nn.logsumexp(history_state.log_weights)

    history_particles = HistoryParticles(observations, actions, carry, log_probs)
    history_state = HistoryState(
        particles=history_particles,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_idx,
        rewards=rewards,
    )
    return history_state, belief_state, belief_info, log_marginal


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9))
def smc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_history_particles: int,
    num_belief_particles: int,
    prior_dist: Distribution,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
    resample_fn: Callable = systematic_resampling
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    """
    Perform the Sequential Monte Carlo (SMC) algorithm.

    Args:
        rng_key: PRNGKey
            The random key for sampling.
        num_history_particles: int
            The number of history particles.
        num_belief_particles: int
            The number of belief particles.
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
        params: Parameters
            Parameters of the recurrent policy.
        reward_fn: RewardFn
            The reward function.
        tempering: float
            The tempering parameter.
        slew_rate_penalty: float
            The slew rate penalty.
        resample_fn: Callable
            The resampling function.

    Returns:
        tuple[HistoryState, BeliefState, Array]
            All history and belief states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def smc_loop(carry: tuple[HistoryState, BeliefState, Array], args: tuple[int, PRNGKey]):
        history_state, belief_state, log_marginal = carry
        time_idx, key = args

        history_state, belief_state, belief_info, log_marginal_incr = smc_step(
            time_idx,
            key,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty,
            resample_fn,
            history_state,
            belief_state,
        )

        log_marginal += log_marginal_incr
        return (history_state, belief_state, log_marginal), (history_state, belief_state, belief_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_history_state, init_belief_state, init_belief_info = smc_init(
        init_key,
        num_history_particles,
        num_belief_particles,
        prior_dist,
        obs_model,
        policy,
        params,
    )

    time_indices = jnp.arange(1, num_time_steps + 1)
    keys = random.split(loop_key, num_time_steps)
    (_, _, log_marginal), (history_states, belief_states, belief_infos) = jax.lax.scan(
        smc_loop,
        (init_history_state, init_belief_state, jnp.array(0.0)),
        (time_indices, keys),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    history_states = concat_trees(init_history_state, history_states)
    belief_states = concat_trees(init_belief_state, belief_states)
    belief_infos = concat_trees(init_belief_info, belief_infos)
    return history_states, belief_states, belief_infos, log_marginal


@partial(jax.jit, static_argnums=(5,))
def backward_tracing(
    rng_key: PRNGKey,
    history_states: HistoryState,
    belief_states: BeliefState,
    belief_infos: BeliefInfo,
    resample: bool = True,
    resample_fn: Callable = systematic_resampling,
) -> tuple[HistoryParticles, BeliefState, BeliefInfo]:
    """Genealogy tracking to get the smoothed trajectories.

    Args:
        rng_key: The random number generator key.
        history_states: The history states of the history particle filter.
        belief_states: The belief states of the belief particle filter.
        belief_infos: The belief infos of the belief particle filter.
        resample: If True, sample the genealogy, otherwise trace back all final
          particles.
        resample_fn: The resampling function.

    Returns:
        The traced history and belief states.
    """
    _, num_particles = history_states.weights.shape

    # Sample the states at the final time step.
    resampling_idx = jax.lax.select(
        resample,
        resample_fn(rng_key, history_states.weights[-1], num_particles),
        jnp.arange(num_particles, dtype=jnp.int32),
    )

    last_history_particles = jax.tree.map(lambda x: x[-1, resampling_idx], history_states.particles)
    last_belief_state = jax.tree.map(lambda x: x[-1, resampling_idx], belief_states)
    last_belief_info = jax.tree.map(lambda x: x[-1, resampling_idx], belief_infos)

    # Trace the genealogy for the history states.
    def tracing_fn(carry, args):
        idx = carry
        particles, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = jax.tree.map(lambda x: x[a], particles)
        return a, (a, ancestors)

    _, (traced_indices, traced_history_particles) = jax.lax.scan(
        tracing_fn,
        resampling_idx,
        (
            jax.tree.map(lambda x: x[:-1], history_states.particles),
            history_states.resampling_indices[1:],
        ),
        reverse=True,
    )

    # Trace the belief states.^
    def get_traced_belief(idx, state, info):
        return jax.tree.map(lambda x: x[idx], state), jax.tree.map(lambda x: x[idx], info)

    traced_belief_states, traced_belief_infos = jax.vmap(get_traced_belief, in_axes=(0, 0, 0))(
        traced_indices,
        jax.tree.map(lambda x: x[:-1], belief_states),
        jax.tree.map(lambda x: x[:-1], belief_infos),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    traced_history_particles = concat_trees(traced_history_particles, last_history_particles)
    traced_belief_states = concat_trees(traced_belief_states, last_belief_state)
    traced_belief_infos = concat_trees(traced_belief_infos, last_belief_info)
    return traced_history_particles, traced_belief_states, traced_belief_infos


def mcmc_backward_sampling_single(
    rng_key: PRNGKey,
    history_states: HistoryState,
    belief_states: BeliefState,
    trans_model: TransitionModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
) -> tuple[HistoryParticles, HistoryState]:
    """
    MCMC-based backward sampling from Bunch and Godsill (2013)
    """
    num_time_steps, num_history_particles = history_states.weights.shape

    _, _, action_dim = history_states.particles.actions.shape
    _, _, observation_dim = history_states.particles.observations.shape

    # Sample the last particle
    key, sub_key = random.split(rng_key)
    idx = jax.random.choice(
        sub_key, jnp.arange(num_history_particles, dtype=jnp.int32), p=history_states.weights[-1]
    )
    last_smoothed_history_particles = jax.tree.map(lambda x: x[-1, idx], history_states.particles)
    last_smoothed_belief_state = jax.tree.map(lambda x: x[-1, idx], belief_states)

    smoothed_actions = jnp.zeros((num_time_steps, action_dim))
    smoothed_observations = jnp.zeros((num_time_steps, observation_dim))

    smoothed_actions = smoothed_actions.at[-1, :].set(last_smoothed_history_particles.actions)
    smoothed_observations = smoothed_observations.at[-1, :].set(last_smoothed_history_particles.observations)

    def body(carry, args):
        idx, smoothed_actions, smoothed_observations = carry
        t, key = args

        ancestor_idx = history_states.resampling_indices[t + 1, idx]
        key, sub_key = random.split(key)
        proposed_idx = jax.random.choice(
            sub_key, jnp.arange(num_history_particles, dtype=jnp.int32), p=history_states.weights[t]
        )

        ancestor_policy_logpdf = policy_logpdf(
            t + 1,  # starting index of future time steps
            smoothed_actions,
            smoothed_observations,
            jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.carry),
            jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.observations),
            policy,
            params
        )
        proposed_policy_logpdf = policy_logpdf(
            t + 1,  # starting index of future time steps
            smoothed_actions,
            smoothed_observations,
            jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.carry),
            jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.observations),
            policy,
            params
        )

        ancestor_trans_logpdf = transition_logpdf(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            jax.tree.map(lambda x: x[t, ancestor_idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            trans_model,
        )
        proposed_trans_logpdf = transition_logpdf(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            jax.tree.map(lambda x: x[t, proposed_idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            trans_model,
        )

        ancestor_log_potential, _ = log_potential(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            history_states.particles.actions[t, ancestor_idx],
            t + 1,
            reward_fn,
            tempering,
            slew_rate_penalty
        )
        proposed_log_potential, _ = log_potential(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            history_states.particles.actions[t, proposed_idx],
            t + 1,
            reward_fn,
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

        smoothed_history_particles = jax.tree.map(lambda x: x[t, idx], history_states.particles)
        smoothed_belief_state = jax.tree.map(lambda x: x[t, idx], belief_states)

        smoothed_actions = smoothed_actions.at[t, :].set(smoothed_history_particles.actions)
        smoothed_observations = smoothed_observations.at[t, :].set(smoothed_history_particles.observations)
        return (idx, smoothed_actions, smoothed_observations), \
            (smoothed_history_particles, smoothed_belief_state)

    _, (sampled_history_states, sampled_belief_states) = jax.lax.scan(
        body,
        (idx, smoothed_actions, smoothed_observations),
        (
            jnp.arange(num_time_steps - 1),
            random.split(key, num_time_steps - 1)
        ), reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    smoothed_history_particles = concat_trees(sampled_history_states, last_smoothed_history_particles)
    smoothed_belief_states = concat_trees(sampled_belief_states, last_smoothed_belief_state)
    return smoothed_history_particles, smoothed_belief_states


@partial(jax.jit, static_argnums=(1, 4, 5, 7, 8, 9))
def mcmc_backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    history_states: HistoryState,
    belief_states: BeliefState,
    trans_model: TransitionModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float = 0.0,
):
    keys = random.split(rng_key, num_samples)
    smoothed_history_particles, smoothed_belief_states = jax.vmap(
        lambda key: mcmc_backward_sampling_single(
            key,
            history_states,
            belief_states,
            trans_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty),
        out_axes=1)(keys)
    return smoothed_history_particles, smoothed_belief_states


def backward_sampling_single(
    rng_key: PRNGKey,
    history_states: HistoryState,
    belief_states: BeliefState,
    trans_model: TransitionModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
) -> tuple[HistoryParticles, BeliefState]:
    num_time_steps, num_history_particles = history_states.weights.shape

    _, _, action_dim = history_states.particles.actions.shape
    _, _, observation_dim = history_states.particles.observations.shape

    # Sample the last particle
    key, sub_key = random.split(rng_key)
    idx = jax.random.choice(
        sub_key, jnp.arange(num_history_particles, dtype=jnp.int32), p=history_states.weights[-1]
    )
    last_smoothed_history_particles = jax.tree.map(lambda x: x[-1, idx], history_states.particles)
    last_smoothed_belief_state = jax.tree.map(lambda x: x[-1, idx], belief_states)

    smoothed_actions = jnp.zeros((num_time_steps, action_dim))
    smoothed_observations = jnp.zeros((num_time_steps, observation_dim))

    smoothed_actions = smoothed_actions.at[-1, :].set(last_smoothed_history_particles.actions)
    smoothed_observations = smoothed_observations.at[-1, :].set(last_smoothed_history_particles.observations)

    vmap_policy_logpdf = jax.vmap(policy_logpdf, in_axes=(None, None, None, 0, 0, None, None))
    vmap_transition_logpdf = jax.vmap(transition_logpdf, in_axes=(None, 0, None, None))
    vmap_log_potential = jax.vmap(log_potential, in_axes=(None, None, 0, None, None, None, None))

    def body(carry, args):
        idx, smoothed_actions, smoothed_observations = carry
        t, key = args

        log_policy = vmap_policy_logpdf(
            t + 1,  # starting index of future time steps
            smoothed_actions,
            smoothed_observations,
            jax.tree.map(lambda x: x[t, :], history_states.particles.carry),
            jax.tree.map(lambda x: x[t, :], history_states.particles.observations),
            policy,
            params
        )

        log_transition = vmap_transition_logpdf(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            jax.tree.map(lambda x: x[t, :], belief_states),
            history_states.particles.actions[t + 1, idx],
            trans_model,
        )

        log_potentials, _ = vmap_log_potential(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            history_states.particles.actions[t, :],
            t + 1,
            reward_fn,
            tempering,
            slew_rate_penalty
        )

        reweighting_ratio = log_policy + log_transition + log_potentials
        smoothing_weights = jax.nn.softmax(reweighting_ratio + history_states.log_weights[t])

        idx = jax.random.choice(sub_key, jnp.arange(num_history_particles, dtype=jnp.int32), p=smoothing_weights)
        smoothed_history_particles = jax.tree.map(lambda x: x[t, idx], history_states.particles)
        smoothed_belief_state = jax.tree.map(lambda x: x[t, idx], belief_states)

        smoothed_actions = smoothed_actions.at[t, :].set(smoothed_history_particles.actions)
        smoothed_observations = smoothed_observations.at[t, :].set(smoothed_history_particles.observations)
        return (idx, smoothed_actions, smoothed_observations), \
            (smoothed_history_particles, smoothed_belief_state)

    _, (sampled_history_states, sampled_belief_states) = jax.lax.scan(
        body,
        (idx, smoothed_actions, smoothed_observations),
        (
            jnp.arange(num_time_steps - 1),
            random.split(key, num_time_steps - 1)
        ), reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    smoothed_history_particles = concat_trees(sampled_history_states, last_smoothed_history_particles)
    smoothed_belief_states = concat_trees(sampled_belief_states, last_smoothed_belief_state)
    return smoothed_history_particles, smoothed_belief_states


@partial(jax.jit, static_argnums=(1, 4, 5, 7, 8, 9))
def backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    history_state: HistoryState,
    belief_state: BeliefState,
    trans_model: TransitionModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float = 0.0,
):
    keys = random.split(rng_key, num_samples)
    smoothed_history_particles, smoothed_belief_states = jax.vmap(
        lambda key: backward_sampling_single(
            key,
            history_state,
            belief_state,
            trans_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty),
        out_axes=1)(keys)
    return smoothed_history_particles, smoothed_belief_states
