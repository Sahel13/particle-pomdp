from functools import partial
from typing import Callable

import jax
from jax import random, numpy as jnp, Array
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
    custom_split
)


def rsmc_init(
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

    # Initialize dummy actions and policy carry.
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


def rsmc_step(
    rng_key: PRNGKey,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    policy_posterior: RecurrentPolicy,
    policy_posterior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    damping: float,
    resample_fn: Callable,
    history_state: HistoryState,
    belief_state: BeliefState,
    time_idx: int
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    r"""Performs one iteration of the regularized nested Sequential Monte Carlo (SMC) algorithm.

    This function implements a single step of the regularized nested SMC algorithm. It updates
    both history and belief states using importance sampling with a regularization term that
    incorporates information from both prior and posterior policies.

    The algorithm consists of several key steps:
    1. Resampling history particles
    2. Resampling belief particles
    3. Sampling new actions using both prior and posterior policies
    4. Propagating belief particles through the transition model
    5. Sampling new observations and updating particle weights
    6. Computing marginal likelihood increments

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling operations.
        policy_prior: RecurrentPolicy
            Prior stochastic policy π_prior used for proposing actions.
        policy_prior_params: Parameters
            Parameters of the prior policy.
        policy_posterior: RecurrentPolicy
            Posterior stochastic policy π_post used for regularization.
        policy_posterior_params: Parameters
            Parameters of the posterior policy.
        trans_model: TransitionModel
            The prior transition model for the state, $f(s_t \mid s_{t-1}, a_{t-1})$.
        obs_model: ObservationModel
            Observation model p(z_t | s_t).
        reward_fn: RewardFn
            Function computing rewards r(s_t, a_t).
        slew_rate_penalty: float
            Penalty coefficient for rapid action changes.
        tempering: float
            Temperature parameter η controlling the influence of the reward.
        damping: float
            Damping parameter β balancing prior and posterior policies.
        resample_fn: Callable
            Function implementing the resampling strategy.
        history_state: HistoryState
            Current state of history particles (actions, observations, weights).
        belief_state: BeliefState
            Current state of belief particles (states, weights).
        time_idx: int
            Current timestep in the sequence.

    Returns:
        tuple[HistoryState, BeliefState, BeliefInfo, Array]:
            - Updated history state containing new particles and weights
            - Updated belief state with propagated particles and weights
            - Belief state statistics (ESS, mean, covariance)
            - Log marginal likelihood increment for this step

    Notes:
        The regularization term is controlled by the damping parameter β:
        log w = (1-β)log p(r) - β KL(π_post || π_prior)
        where p(r) is the reward likelihood and KL is the policy divergence.
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
    carry, actions, actions_prior_log_prob, _ = policy_prior.sample_and_log_prob(
        action_key,
        particles.carry,
        particles.actions,
        particles.observations,
        policy_prior_params
    )
    actions_post_log_prob = policy_posterior.log_prob(
        actions,
        particles.carry,
        particles.actions,
        particles.observations,
        policy_posterior_params,
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

    log_weights = history_state.log_weights \
                  + (1. - damping) * log_potentials \
                  - damping * actions_prior_log_prob \
                  + damping * actions_post_log_prob \

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
        "policy_posterior",
        "trans_model",
        "obs_model",
        "reward_fn"
    )
)
def rsmc(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_history_particles: int,
    num_belief_particles: int,
    init_prior: Distribution,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    policy_posterior: RecurrentPolicy,
    policy_posterior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    damping: float,
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
        policy_prior: RecurrentPolicy
            The prior recurrent policy.
        policy_prior_params: Parameters
            Parameters of the prior recurrent policy.
        policy_posterior: RecurrentPolicy
            The posterior recurrent policy.
        policy_posterior_params: Parameters
            Parameters of the posterior recurrent policy.
        trans_model: TransitionModel
            The transition model for the state.
        obs_model: ObservationModel
            The observation model.
        reward_fn: RewardFn
            The reward function.
        slew_rate_penalty: float
            The slew rate penalty.
        tempering: float
            The tempering parameter.
        damping: float
            The damping parameter.
        resample_fn: Callable
            The resampling function.

    Returns:
        tuple[HistoryState, BeliefState, Array]
            All history and belief states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def rsmc_loop(carry, args):
        history_state, belief_state, log_marginal = carry
        time_idx, key = args

        history_state, belief_state, belief_info, log_marginal_incr = \
            rsmc_step(
                rng_key=key,
                policy_prior=policy_prior,
                policy_prior_params=policy_prior_params,
                policy_posterior=policy_posterior,
                policy_posterior_params=policy_posterior_params,
                trans_model=trans_model,
                obs_model=obs_model,
                reward_fn=reward_fn,
                slew_rate_penalty=slew_rate_penalty,
                tempering=tempering,
                damping=damping,
                resample_fn=resample_fn,
                history_state=history_state,
                belief_state=belief_state,
                time_idx=time_idx
            )

        log_marginal += log_marginal_incr
        return (history_state, belief_state, log_marginal), \
            (history_state, belief_state, belief_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_history_state, init_belief_state, init_belief_info = \
        rsmc_init(
            rng_key=init_key,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            init_prior=init_prior,
            policy_prior=policy_prior,
            obs_model=obs_model
        )

    (_, _, log_marginal), (history_states, belief_states, belief_infos) = jax.lax.scan(
        f=rsmc_loop,
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
