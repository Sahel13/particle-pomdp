from functools import partial
from typing import Callable

import jax
from jax import Array, random, numpy as jnp
from distrax import Distribution

from ppomdp.core import (
    PRNGKey,
    Parameters,
    BeliefState,
    BeliefInfo,
    HistoryState,
    HistoryParticles,
    TransitionModel,
    ObservationModel,
    RewardFn,
    RecurrentPolicy,
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
    custom_split
)


def smc_init(
    rng_key: PRNGKey,
    num_history_particles: int,
    num_belief_particles: int,
    init_prior: Distribution,
    policy_prior: RecurrentPolicy,
    obs_model: ObservationModel
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
        init_prior: distrax.Distribution
            The prior distribution for the initial state particles.
        policy_prior: RecurrentPolicy
            The recurrent policy.
        obs_model: ObservationModel
            The observation model.
    """
    key, sub_key = random.split(rng_key)
    belief_particles = init_prior.sample(
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

    # Initialize dummy actions, encodings and policy carry.
    init_carry = policy_prior.reset(num_history_particles)
    dummy_actions = jnp.zeros((num_history_particles, policy_prior.dim))

    history_particles = HistoryParticles(
        actions=dummy_actions,
        carry=init_carry,
        observations=observations,
    )
    history_state = HistoryState(
        particles=history_particles,
        log_weights=jnp.zeros(num_history_particles),
        weights=jnp.ones(num_history_particles) / num_history_particles,
        resampling_indices=jnp.zeros(num_history_particles, dtype=jnp.int32),
        rewards=jnp.zeros(num_history_particles),
    )

    return history_state, belief_state, belief_info


def smc_step(
    rng_key: PRNGKey,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    resample_fn: Callable,
    history_state: HistoryState,
    belief_state: BeliefState,
    time_idx: int
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    r"""A single step of the nested SMC algorithm.

    Args:
        rng_key: PRNGKey
        trans_model: TransitionModel
            The transition model for the state, $f(s_t \mid s_{t-1}, a_{t-1})$.
        policy_prior: RecurrentPolicy
            The stochastic policy, $\pi_\phi$.
        policy_prior_params: Parameters
            Parameters of recurrent policy $\phi$.
        obs_model: ObservationModel
            The observation model, $g(z_t \mid s_t)$.
        reward_fn: RewardFn
            The reward function, $r(s_t, a_t)$.
        slew_rate_penalty: float
            The slew rate penalty.
        tempering: float
            The tempering parameter, $\eta$.
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
    key, sub_keys = custom_split(key, num_particles + 1)
    belief_state = jax.vmap(resample_belief, in_axes=(0, 0, None))(
        sub_keys, belief_state, resample_fn
    )

    # 3. Sample new actions.
    key, action_key = random.split(key)
    carry, actions, _ = policy_prior.sample(
        action_key, particles.carry, particles.observations, particles.actions, policy_prior_params
    )

    # 4. Propagate the belief particles.
    key, sub_keys = custom_split(key, num_particles + 1)
    belief_particles = jax.vmap(propagate_belief, in_axes=(0, None, 0, 0))(
        sub_keys, trans_model, belief_state.particles, actions
    )
    belief_state = belief_state._replace(particles=belief_particles)

    # 5. Sample new observations.
    key, sub_keys = custom_split(key, num_particles + 1)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        sub_keys, obs_model, belief_state
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
    log_potentials, rewards = jax.vmap(
        log_potential, in_axes=(0, 0, 0, None, None, None, None))(
        belief_state, actions, particles.actions, time_idx, reward_fn, slew_rate_penalty, tempering
    )

    log_weights = history_state.log_weights + log_potentials
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jax.nn.softmax(log_weights)

    # 8. Compute the normalizing constant increment.
    # Eq. 10.3 in Chopin and Papaspiliopoulos (2020).
    log_marginal = logsum_weights - jax.nn.logsumexp(history_state.log_weights)

    history_particles = HistoryParticles(
        actions=actions,
        carry=carry,
        observations=observations,
    )
    history_state = HistoryState(
        particles=history_particles,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_idx,
        rewards=rewards,
    )
    return history_state, belief_state, belief_info, log_marginal


@partial(
    jax.jit,
    static_argnames=(
        "num_time_steps",
        "num_history_particles",
        "num_belief_particles",
        "init_prior",
        "policy_prior",
        "trans_model",
        "obs_model",
        "reward_fn"
    )
)
def smc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_history_particles: int,
    num_belief_particles: int,
    init_prior: Distribution,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
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
        init_prior: Distribution
            The prior distribution for the initial state particles.
        trans_model: TransitionModel
            The transition model for the state.
        policy_prior: RecurrentPolicy
            The recurrent policy.
        policy_prior_params: Parameters
            Parameters of the recurrent policy.
        obs_model: ObservationModel
            The observation model.
        reward_fn: RewardFn
            The reward function.
        slew_rate_penalty: float
            The slew rate penalty.
        tempering: float
            The tempering parameter.
        resample_fn: Callable
            The resampling function.

    Returns:
        tuple[HistoryState, BeliefState, Array]
            All history and belief states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def smc_loop(carry, args):
        history_state, belief_state, log_marginal = carry
        time_idx, key = args

        history_state, belief_state, belief_info, log_marginal_incr = \
            smc_step(
                rng_key=key,
                policy_prior=policy_prior,
                policy_prior_params=policy_prior_params,
                trans_model=trans_model,
                obs_model=obs_model,
                reward_fn=reward_fn,
                slew_rate_penalty=slew_rate_penalty,
                tempering=tempering,
                resample_fn=resample_fn,
                history_state=history_state,
                belief_state=belief_state,
                time_idx=time_idx,
            )

        log_marginal += log_marginal_incr
        return (history_state, belief_state, log_marginal), (history_state, belief_state, belief_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_history_state, init_belief_state, init_belief_info = \
        smc_init(
            rng_key=init_key,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            init_prior=init_prior,
            policy_prior=policy_prior,
            obs_model=obs_model
        )

    (_, _, log_marginal), (history_states, belief_states, belief_infos) = jax.lax.scan(
        f=smc_loop,
        init=(
            init_history_state,
            init_belief_state,
            jnp.array(0.0)
        ),
        xs=(
            jnp.arange(1, num_time_steps + 1),
            random.split(loop_key, num_time_steps)
        )
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    history_states = concat_trees(init_history_state, history_states)
    belief_states = concat_trees(init_belief_state, belief_states)
    belief_infos = concat_trees(init_belief_info, belief_infos)
    return history_states, belief_states, belief_infos, log_marginal


@partial(jax.jit, static_argnames=("resample_fn",))
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
        f=tracing_fn,
        init=resampling_idx,
        xs=(
            jax.tree.map(lambda x: x[:-1], history_states.particles),
            history_states.resampling_indices[1:],
        ),
        reverse=True,
    )

    # Trace the belief states.
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
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float
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
        key=sub_key,
        a=jnp.arange(num_history_particles, dtype=jnp.int32),
        p=history_states.weights[-1]
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
            sub_key,
            jnp.arange(num_history_particles, dtype=jnp.int32),
            p=history_states.weights[t]
        )
        ancestor_policy_logpdf = policy_logpdf(
            t + 1,  # starting index of future time steps
            smoothed_actions,
            smoothed_observations,
            jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.carry),
            jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.observations),
            policy_prior,
            policy_prior_params
        )
        proposed_policy_logpdf = policy_logpdf(
            t + 1,  # starting index of future time steps
            smoothed_actions,
            smoothed_observations,
            jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.carry),
            jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.observations),
            policy_prior,
            policy_prior_params
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
            slew_rate_penalty,
            tempering,
        )
        proposed_log_potential, _ = log_potential(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            history_states.particles.actions[t, proposed_idx],
            t + 1,
            reward_fn,
            slew_rate_penalty,
            tempering,
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
        f=body,
        init=(idx, smoothed_actions, smoothed_observations),
        xs=(
            jnp.arange(num_time_steps - 1),
            random.split(key, num_time_steps - 1)
        ),
        reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    smoothed_history_particles = concat_trees(sampled_history_states, last_smoothed_history_particles)
    smoothed_belief_states = concat_trees(sampled_belief_states, last_smoothed_belief_state)
    return smoothed_history_particles, smoothed_belief_states


@partial(
    jax.jit,
    static_argnames=(
        "num_samples",
        "trans_model",
        "policy_prior",
        "reward_fn",
        "slew_rate_penalty",
        "tempering",
    )
)
def mcmc_backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    history_states: HistoryState,
    belief_states: BeliefState,
    trans_model: TransitionModel,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float
):
    keys = random.split(rng_key, num_samples)
    smoothed_history_particles, smoothed_belief_states = jax.vmap(
        lambda key: mcmc_backward_sampling_single(
            key,
            history_states,
            belief_states,
            trans_model,
            policy_prior,
            policy_prior_params,
            reward_fn,
            slew_rate_penalty,
            tempering
        ), out_axes=1)(keys)
    return smoothed_history_particles, smoothed_belief_states


def backward_sampling_single(
    rng_key: PRNGKey,
    history_states: HistoryState,
    belief_states: BeliefState,
    trans_model: TransitionModel,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float
) -> tuple[HistoryParticles, BeliefState]:
    """Performs backward sampling for a single trajectory using the forward filter-backward sampler algorithm.

    This implementation follows the forward filtering-backward sampling (FFBS) approach to generate
    smoothed trajectories from Sequential Monte Carlo results. It samples backwards through time,
    selecting particles according to their smoothing weights which account for both past and future information.

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling operations.
        history_states: HistoryState
            Forward filter history states containing particles and weights.
        belief_states: BeliefState
            Forward filter belief states containing state particles and weights.
        trans_model: TransitionModel
            State transition model p(s_t | s_{t-1}, a_{t-1}).
        policy_prior: RecurrentPolicy
            Prior policy model used in the forward pass.
        policy_prior_params: Parameters
            Parameters of the prior policy.
        reward_fn: RewardFn
            Function computing rewards r(s_t, a_t).
        slew_rate_penalty: float
            Penalty coefficient for rapid action changes.
        tempering: float
            Temperature parameter controlling the influence of rewards.

    Returns:
        tuple[HistoryParticles, BeliefState]:
            - Smoothed history particles containing the sampled trajectory
            - Associated belief states for the sampled trajectory

    Notes:
        The smoothing weights are computed using:
        w_smooth ∝ w_filter * p(x_{t+1} | x_t) * π(a_{t+1} | h_t)
        where w_filter are the filtering weights from the forward pass.
    """
    num_time_steps, num_history_particles = history_states.weights.shape

    _, _, action_dim = history_states.particles.actions.shape
    _, _, observation_dim = history_states.particles.observations.shape

    # Sample the last particle
    key, sub_key = random.split(rng_key)
    idx = jax.random.choice(
        key=sub_key,
        a=jnp.arange(num_history_particles, dtype=jnp.int32),
        p=history_states.weights[-1]
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
            policy_prior,
            policy_prior_params
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
            slew_rate_penalty,
            tempering,
        )

        reweighting_ratio = log_policy + log_transition + log_potentials
        smoothing_weights = jax.nn.softmax(reweighting_ratio + history_states.log_weights[t])

        key, sub_key = random.split(key)
        idx = jax.random.choice(sub_key, jnp.arange(num_history_particles, dtype=jnp.int32), p=smoothing_weights)
        smoothed_history_particles = jax.tree.map(lambda x: x[t, idx], history_states.particles)
        smoothed_belief_state = jax.tree.map(lambda x: x[t, idx], belief_states)

        smoothed_actions = smoothed_actions.at[t, :].set(smoothed_history_particles.actions)
        smoothed_observations = smoothed_observations.at[t, :].set(smoothed_history_particles.observations)
        return (idx, smoothed_actions, smoothed_observations), \
            (smoothed_history_particles, smoothed_belief_state)

    _, (sampled_history_states, sampled_belief_states) = jax.lax.scan(
        f=body,
        init=(idx, smoothed_actions, smoothed_observations),
        xs=(
            jnp.arange(num_time_steps - 1),
            random.split(key, num_time_steps - 1)
        ),
        reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    smoothed_history_particles = concat_trees(sampled_history_states, last_smoothed_history_particles)
    smoothed_belief_states = concat_trees(sampled_belief_states, last_smoothed_belief_state)
    return smoothed_history_particles, smoothed_belief_states


@partial(
    jax.jit,
    static_argnames=(
        "num_samples",
        "trans_model",
        "policy_prior",
        "reward_fn",
        "slew_rate_penalty",
        "tempering"
    )
)
def backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    history_state: HistoryState,
    belief_state: BeliefState,
    trans_model: TransitionModel,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float
) -> tuple[HistoryParticles, BeliefState]:
    """Generate multiple smoothed trajectories using backward sampling.

    This function generates multiple independent smoothed trajectories by applying
    backward sampling repeatedly. It uses JAX's vectorization capabilities to
    efficiently generate multiple trajectories in parallel.

    Args:
        rng_key: PRNGKey
            Random number generator key.
        num_samples: int
            Number of independent trajectories to generate.
        history_state: HistoryState
            Forward filter history states.
        belief_state: BeliefState
            Forward filter belief states.
        trans_model: TransitionModel
            State transition model.
        policy_prior: RecurrentPolicy
            Prior policy used in forward filtering.
        policy_prior_params: Parameters
            Parameters of the prior policy.
        reward_fn: RewardFn
            Reward function.
        slew_rate_penalty: float
            Action smoothness penalty coefficient.
        tempering: float
            Temperature parameter for reward weighting.

    Returns:
        tuple[HistoryParticles, BeliefState]:
            - Batch of smoothed history particles, shape (T, num_samples, ...)
            - Batch of smoothed belief states, shape (T, num_samples, ...)
    """
    keys = random.split(rng_key, num_samples)
    smoothed_history_particles, smoothed_belief_states = jax.vmap(
        lambda key: backward_sampling_single(
            key,
            history_state,
            belief_state,
            trans_model,
            policy_prior,
            policy_prior_params,
            reward_fn,
            slew_rate_penalty,
            tempering
        ), out_axes=1)(keys)
    return smoothed_history_particles, smoothed_belief_states
