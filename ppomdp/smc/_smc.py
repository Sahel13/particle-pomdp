from functools import partial
from typing import Callable, Union

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
    AttentionPolicy,
)
from ppomdp.utils import custom_split
from ppomdp.smc.utils import (
    resample_belief,
    propagate_belief,
    reweight_belief,
    sample_marginal_obs,
    log_potential,
    resample_history,
    effective_sample_size,
    systematic_resampling,
    action_sequence_log_prob,
    transition_marginal_log_prob,
    weighted_mean,
    weighted_covar,
)


def smc_init(
    rng_key: PRNGKey,
    num_history_particles: int,
    num_belief_particles: int,
    belief_prior: Distribution,
    policy_prior: Union[RecurrentPolicy, AttentionPolicy],
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
        belief_prior: distrax.Distribution
            The prior distribution for the initial state particles.
        policy_prior: RecurrentPolicy
            The recurrent policy.
        obs_model: ObservationModel
            The observation model.
    """
    key, sub_key = random.split(rng_key)
    belief_particles = belief_prior.sample(
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
    policy_prior: Union[RecurrentPolicy, AttentionPolicy],
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    history_resample_fn: Callable,
    belief_resample_fn: Callable,
    history_state: HistoryState,
    belief_state: BeliefState,
    time_idx: int
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    r"""A single step of the nested SMC algorithm.

    Args:
        rng_key: PRNGKey
        trans_model: TransitionModel
            The transition model for the state, $f(s_t \mid s_{t-1}, a_{t-1})$.
        policy_prior: Union[RecurrentPolicy, AttentionPolicy]
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
        history_resample_fn: Callable
            The resampling function for the history particles.
        belief_resample_fn: Callable
            The resampling function for the belief particles.
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
    history_state = resample_history(sub_key, history_state, history_resample_fn)
    particles = history_state.particles
    resampling_idx = history_state.resampling_indices
    belief_state = jax.tree.map(lambda x: x[resampling_idx], belief_state)

    # 2. Resample the belief particles.
    key, sub_keys = custom_split(key, num_particles + 1)
    belief_state = jax.vmap(resample_belief, in_axes=(0, 0, None))(
        sub_keys, belief_state, belief_resample_fn
    )

    # 3. Sample new actions.
    key, action_key = random.split(key)
    if isinstance(policy_prior, RecurrentPolicy):
        carry, actions, _ = policy_prior.sample(
            rng_key=action_key,
            carry=particles.carry,
            actions=particles.actions,
            observations=particles.observations,
            params=policy_prior_params
        )
    else:  # isinstance(AttentionPolicy)
        carry = None
        actions, _ = policy_prior.sample(
            rng_key=action_key,
            particles=belief_state.particles,
            weights=belief_state.weights,
            params=policy_prior_params
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
        belief_state,
        actions,
        particles.actions,
        time_idx,
        reward_fn,
        slew_rate_penalty,
        tempering
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
        "belief_prior",
        "policy_prior",
        "trans_model",
        "obs_model",
        "reward_fn",
        "history_resample_fn",
        "belief_resample_fn"
    )
)
def smc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_history_particles: int,
    num_belief_particles: int,
    belief_prior: Distribution,
    policy_prior: Union[RecurrentPolicy, AttentionPolicy],
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    history_resample_fn: Callable = systematic_resampling,
    belief_resample_fn: Callable = systematic_resampling,
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
        belief_prior: Distribution
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
        history_resample_fn: Callable
            The resampling function for the history particles.
        belief_resample_fn: Callable
            The resampling function for the belief particles.

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
                history_resample_fn=history_resample_fn,
                belief_resample_fn=belief_resample_fn,
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
            belief_prior=belief_prior,
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
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    history_states: HistoryState,
    belief_states: BeliefState,
) -> tuple[HistoryParticles, HistoryState]:
    r"""Perform a single backward pass of MCMC-based backward sampling.

    This function implements one iteration of the MCMC backward sampling algorithm
    proposed by Bunch and Godsill (2013). It samples a single smoothed trajectory
    by moving backward in time and, at each step `t`, proposing a particle index
    from the filtered distribution at time `t` and accepting/rejecting it based
    on a Metropolis-Hastings ratio involving the policy, transition, and potential
    terms.

    Args:
        rng_key: PRNGKey
            Random number generator key.
        policy_prior: RecurrentPolicy
            The policy model used during the forward pass.
        policy_prior_params: Parameters
            Parameters of the policy prior.
        trans_model: TransitionModel
            The state transition model.
        reward_fn: RewardFn
            The reward function (used within `log_potential`).
        slew_rate_penalty: float
            Penalty for action changes (used within `log_potential`).
        tempering: float
            Tempering factor (used within `log_potential`).
        history_states: HistoryState
            History states obtained from the forward SMC pass (shape T+1, N, ...).
        belief_states: BeliefState
            Belief states obtained from the forward SMC pass (shape T+1, N, M, ...).

    Returns:
        tuple[HistoryParticles, BeliefState]:
            - HistoryParticles: The single smoothed history trajectory (shape T+1, ...).
            - BeliefState: The corresponding smoothed belief states (shape T+1, M, ...).
    """
    num_time_steps_plus_one, num_history_particles = history_states.weights.shape

    _, _, action_dim = history_states.particles.actions.shape
    _, _, observation_dim = history_states.particles.observations.shape

    # Sample the last particle
    key, sub_key = random.split(rng_key)
    idx = jax.random.choice(
        key=sub_key,
        a=jnp.arange(num_history_particles, dtype=jnp.int32),
        p=history_states.weights[-1]
    )
    last_history_particles = jax.tree.map(lambda x: x[-1, idx], history_states.particles)
    last_belief_state = jax.tree.map(lambda x: x[-1, idx], belief_states)

    action_sequence = jnp.zeros((num_time_steps_plus_one, action_dim))
    observation_sequence = jnp.zeros((num_time_steps_plus_one, observation_dim))

    action_sequence = action_sequence.at[-1, :].set(last_history_particles.actions)
    observation_sequence = observation_sequence.at[-1, :].set(last_history_particles.observations)

    def body(carry, args):
        idx, actions, observations = carry
        t, key = args

        ancestor_idx = history_states.resampling_indices[t + 1, idx]
        key, sub_key = random.split(key)
        proposed_idx = jax.random.choice(
            key=sub_key,
            a=jnp.arange(num_history_particles, dtype=jnp.int32),
            p=history_states.weights[t]
        )
        # Compute log π(a_{t+1:T} | h_t^{anc}, z_t^{anc}) using ancestor particle info
        ancestor_action_logpdf = action_sequence_log_prob(
            policy=policy_prior,
            policy_params=policy_prior_params,
            actions_sequence=actions,
            observations_sequence=observations,
            init_carry=jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.carry),
            init_action=jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.actions),
            init_observation=jax.tree.map(lambda x: x[t, ancestor_idx], history_states.particles.observations),
            start_time_idx=t + 1,
        )
        # Compute log π(a_{t+1:T} | h_t^{prop}, z_t^{prop}) using proposed particle info
        proposed_action_logpdf = action_sequence_log_prob(
            policy=policy_prior,
            policy_params=policy_prior_params,
            actions_sequence=actions,
            observations_sequence=observations,
            init_carry=jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.carry),
            init_action=jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.actions),
            init_observation=jax.tree.map(lambda x: x[t, proposed_idx], history_states.particles.observations),
            start_time_idx=t + 1,
        )
        # Compute log p(b_{t+1} | b_t^{anc}, a_{t+1}) using ancestor belief
        ancestor_trans_logpdf = transition_marginal_log_prob(
            trans_model=trans_model,
            next_belief_state=jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            belief_state=jax.tree.map(lambda x: x[t, ancestor_idx], belief_states),
            action=history_states.particles.actions[t + 1, idx],
        )
        # Compute log p(b_{t+1} | b_t^{prop}, a_{t+1}) using proposed belief
        proposed_trans_logpdf = transition_marginal_log_prob(
            trans_model=trans_model,
            next_belief_state=jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            belief_state=jax.tree.map(lambda x: x[t, proposed_idx], belief_states),
            action=history_states.particles.actions[t + 1, idx],
        )
        # Compute log g_{t+1}(b_{t+1}, a_{t+1}, a_t^{anc}) using ancestor previous action
        ancestor_log_potential, _ = log_potential(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            history_states.particles.actions[t, ancestor_idx],
            t + 1,
            reward_fn,
            slew_rate_penalty,
            tempering,
        )
        # Compute log g_{t+1}(b_{t+1}, a_{t+1}, a_t^{prop}) using proposed previous action
        proposed_log_potential, _ = log_potential(
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),
            history_states.particles.actions[t + 1, idx],
            history_states.particles.actions[t, proposed_idx],
            t + 1,
            reward_fn,
            slew_rate_penalty,
            tempering,
        )

        log_prob_ancestor = ancestor_action_logpdf \
            + ancestor_trans_logpdf \
            + ancestor_log_potential

        log_prob_proposed = proposed_action_logpdf \
            + proposed_trans_logpdf \
            + proposed_log_potential

        # accept / reject
        key, sub_key = random.split(key)
        log_u = jnp.log(jax.random.uniform(sub_key))
        log_ratio = log_prob_proposed - log_prob_ancestor
        idx = jax.lax.select(log_ratio > log_u, proposed_idx, ancestor_idx)

        sampled_history = jax.tree.map(lambda x: x[t, idx], history_states.particles)
        sampled_belief = jax.tree.map(lambda x: x[t, idx], belief_states)

        actions = actions.at[t, :].set(sampled_history.actions)
        observations = observations.at[t, :].set(sampled_history.observations)
        return (idx, actions, observations), (sampled_history, sampled_belief)

    _, (sampled_history_particles, sampled_belief_states) = jax.lax.scan(
        f=body,
        init=(idx, action_sequence, observation_sequence),
        xs=(
            jnp.arange(num_time_steps_plus_one - 1),
            random.split(key, num_time_steps_plus_one - 1)
        ),
        reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    sampled_history_particles = concat_trees(sampled_history_particles, last_history_particles)
    sampled_belief_states = concat_trees(sampled_belief_states, last_belief_state)
    return sampled_history_particles, sampled_belief_states


@partial(
    jax.jit,
    static_argnames=(
        "num_samples",
        "policy_prior",
        "trans_model",
        "reward_fn",
        "slew_rate_penalty",
        "tempering",
    )
)
def mcmc_backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    history_states: HistoryState,
    belief_states: BeliefState,
):
    keys = random.split(rng_key, num_samples)
    smoothed_history_particles, smoothed_belief_states = jax.vmap(
        lambda key: mcmc_backward_sampling_single(
            key,
            policy_prior,
            policy_prior_params,
            trans_model,
            reward_fn,
            slew_rate_penalty,
            tempering,
            history_states,
            belief_states,
        ), out_axes=1)(keys)
    return smoothed_history_particles, smoothed_belief_states


def backward_sampling_single(
    rng_key: PRNGKey,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    history_states: HistoryState,
    belief_states: BeliefState,
) -> tuple[HistoryParticles, BeliefState]:
    """Performs backward sampling for a single trajectory using the forward filter-backward sampler algorithm.

    This implementation follows the forward filtering-backward sampling (FFBS) approach to generate
    smoothed trajectories from Sequential Monte Carlo results. It samples backwards through time,
    selecting particles according to their smoothing weights which account for both past and future information.

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling operations.
        policy_prior: RecurrentPolicy
            Prior policy model used in the forward pass.
        policy_prior_params: Parameters
            Parameters of the prior policy.
        trans_model: TransitionModel
            Transition model.
        reward_fn: RewardFn
            Function computing rewards r(s_t, a_t).
        slew_rate_penalty: float
            Penalty coefficient for rapid action changes.
        tempering: float
            Temperature parameter controlling the influence of rewards.
        history_states: HistoryState
            Forward filter history states containing particles and weights.
        belief_states: BeliefState
            Forward filter belief states containing state particles and weights.


    Returns:
        tuple[HistoryParticles, BeliefState]:
            - Smoothed history particles containing the sampled trajectory
            - Associated belief states for the sampled trajectory

    Notes:
        The smoothing weights are computed using:
        w_smooth ∝ w_filter * p(x_{t+1} | x_t) * π(a_{t+1} | h_t)
        where w_filter are the filtering weights from the forward pass.
    """
    num_time_steps_plus_one, num_history_particles = history_states.weights.shape

    _, _, action_dim = history_states.particles.actions.shape
    _, _, observation_dim = history_states.particles.observations.shape

    # Sample the last particle
    key, sub_key = random.split(rng_key)
    idx = jax.random.choice(
        key=sub_key,
        a=jnp.arange(num_history_particles, dtype=jnp.int32),
        p=history_states.weights[-1],
    )
    last_history_particles = jax.tree.map(lambda x: x[-1, idx], history_states.particles)
    last_belief_state = jax.tree.map(lambda x: x[-1, idx], belief_states)

    action_sequence = jnp.zeros((num_time_steps_plus_one, action_dim))
    observation_sequence = jnp.zeros((num_time_steps_plus_one, observation_dim))

    action_sequence = action_sequence.at[-1, :].set(last_history_particles.actions)
    observation_sequence = observation_sequence.at[-1, :].set(last_history_particles.observations)

    vmap_compute_sequence_log_prob = jax.vmap(
        action_sequence_log_prob,
        in_axes=(None, None, None, None, 0, 0, 0, None)
    )
    vmap_transition_marginal_log_prob = jax.vmap(
        transition_marginal_log_prob,
        in_axes=(None, None, 0, None)
    )
    vmap_log_potential = jax.vmap(
        log_potential,
        in_axes=(None, None, 0, None, None, None, None)
    )

    def body(carry, args):
        idx, actions, observations = carry
        t, key = args

        # Compute log π(a_{t+1:T} | h_t^m, z_t^m) for all m
        log_policy = vmap_compute_sequence_log_prob(
            policy_prior,                                               # policy
            policy_prior_params,                                        # params
            actions,                                                    # actions_sequence
            observations,                                               # observations_sequence
            jax.tree.map(lambda x: x[t, :], history_states.particles.carry),          # carry_t_minus_1
            jax.tree.map(lambda x: x[t, :], history_states.particles.actions),        # action_t_minus_1
            jax.tree.map(lambda x: x[t, :], history_states.particles.observations), # observation_t_minus_1
            t + 1,                                                      # start_time_idx
        )

        # Compute log p(b_{t+1}^j | b_t^m, a_{t+1}^j) for all m (where j=idx)
        log_transition = vmap_transition_marginal_log_prob(
            trans_model,                                                # model
            jax.tree.map(lambda x: x[t + 1, idx], belief_states),       # next_belief_state
            jax.tree.map(lambda x: x[t, :], belief_states),             # current_belief
            history_states.particles.actions[t + 1, idx],               # action
        )

        # Compute log g_{t+1}(b_{t+1}^j, a_{t+1}^j, a_t^m) for all m (where j=idx)
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
        sampled_history = jax.tree.map(lambda x: x[t, idx], history_states.particles)
        sampled_belief = jax.tree.map(lambda x: x[t, idx], belief_states)

        actions = actions.at[t, :].set(sampled_history.actions)
        observations = observations.at[t, :].set(sampled_history.observations)
        return (idx, actions, observations), (sampled_history, sampled_belief)

    _, (sampled_history_particles, sampled_belief_states) = jax.lax.scan(
        f=body,
        init=(idx, action_sequence, observation_sequence),
        xs=(
            jnp.arange(num_time_steps_plus_one - 1),
            random.split(key, num_time_steps_plus_one - 1)
        ),
        reverse=True,
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]]), x, y)

    sampled_history_particles = concat_trees(sampled_history_particles, last_history_particles)
    sampled_belief_states = concat_trees(sampled_belief_states, last_belief_state)
    return sampled_history_particles, sampled_belief_states


@partial(
    jax.jit,
    static_argnames=(
        "num_samples",
        "policy_prior",
        "trans_model",
        "reward_fn",
        "slew_rate_penalty",
        "tempering"
    )
)
def backward_sampling(
    rng_key: PRNGKey,
    num_samples: int,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    history_states: HistoryState,
    belief_states: BeliefState,
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
        policy_prior: RecurrentPolicy
            Prior policy used in forward filtering.
        policy_prior_params: Parameters
            Parameters of the prior policy.
        trans_model: TransitionModel
            Transition model.
        reward_fn: RewardFn
            Reward function.
        slew_rate_penalty: float
            Action smoothness penalty coefficient.
        tempering: float
            Temperature parameter for reward weighting.
        history_states: HistoryState
            Forward filter history states.
        belief_states: BeliefState
            Forward filter belief states.

    Returns:
        tuple[HistoryParticles, BeliefState]:
            - Batch of smoothed history particles, shape (T, num_samples, ...)
            - Batch of smoothed belief states, shape (T, num_samples, ...)
    """
    keys = random.split(rng_key, num_samples)
    smoothed_history_particles, smoothed_belief_states = jax.vmap(
        lambda key: backward_sampling_single(
            key,
            policy_prior,
            policy_prior_params,
            trans_model,
            reward_fn,
            slew_rate_penalty,
            tempering,
            history_states,
            belief_states,
        ), out_axes=1)(keys)
    return smoothed_history_particles, smoothed_belief_states
