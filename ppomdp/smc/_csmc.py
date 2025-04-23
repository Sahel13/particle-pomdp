from functools import partial

import jax
from jax import Array, random, numpy as jnp
from distrax import Distribution

from ppomdp.core import (
    BeliefState,
    BeliefInfo,
    ObservationModel,
    HistoryParticles,
    HistoryState,
    Reference,
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
    multinomial_resampling,
    systematic_resampling,
    custom_split,
)


def csmc_init(
    rng_key: PRNGKey,
    num_history_particles: int,
    num_belief_particles: int,
    init_prior: Distribution,
    policy_prior: RecurrentPolicy,
    obs_model: ObservationModel,
    reference: Reference,
) -> tuple[HistoryState, BeliefState, BeliefInfo]:
    """
    Initializes the CSMC (Conditional Sequential Monte Carlo) algorithm by setting up the
    necessary belief and history states, with specified priors and reference state.

    The function creates belief particles based on the initial prior distribution and sets
    up the belief state, which is reweighted using marginal observations sampled for each
    history particle. It incorporates reference states into the initialized belief state
    and observations. Additionally, it initializes the history state with dummy actions
    and policy carry, replacing them with reference values where applicable.

    Wraps up by returning the initialized history state, belief state, and belief info
    structure for further iterations or processing.

    Args:
        rng_key: Random number generator key for stochastic operations.
        num_history_particles: Number of particles in the history state.
        num_belief_particles: Number of particles in the belief state.
        init_prior: Initial prior distribution for belief particles.
        policy_prior: Recurrent policy model defining priors over policies.
        obs_model: Observation model for sampling and updating observations.
        reference: Reference state containing pre-defined belief state and
            history particles.

    Returns:
        tuple[HistoryState, BeliefState, BeliefInfo]: A tuple containing the
        initialized history state, belief state, and belief information metrics
        such as effective sample size, mean, and covariance of belief particles.
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

    # replace zeroth belief state with reference belief state
    belief_state = jax.tree.map(
        lambda x, y: x.at[0].set(y), belief_state, reference.belief_state
    )

    # sample marginal observations
    keys = random.split(key, num_history_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, belief_state
    )

    # replace zeroth observations with reference observation
    observations = observations.at[0].set(reference.history_particles.observations)

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

    # replace zeroth carry, action, and log_prob with reference carry, action, and log_prob
    init_carry = jax.tree_map(lambda x, y: x.at[0].set(y), init_carry, reference.history_particles.carry)
    dummy_actions = dummy_actions.at[0].set(reference.history_particles.actions)

    history_particles = HistoryParticles(
        actions=dummy_actions,
        carry=init_carry,
        observations=observations
    )
    history_state = HistoryState(
        particles=history_particles,
        log_weights=jnp.zeros(num_history_particles),
        weights=jnp.ones(num_history_particles) / num_history_particles,
        resampling_indices=jnp.zeros(num_history_particles, dtype=jnp.int32),
        rewards=jnp.zeros(num_history_particles),
    )
    return history_state, belief_state, belief_info


def csmc_step(
    rng_key: PRNGKey,
    policy_prior: RecurrentPolicy,
    policy_prior_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
    reference: Reference,
    history_state: HistoryState,
    belief_state: BeliefState,
    time_idx: int,
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    r"""A single step of the nested CSMC algorithm.

    Args:
        time_idx: int
            The current time index.
        rng_key: PRNGKey
        trans_model: TransitionModel
            The transition model for the state, $f(s_t \mid s_{t-1}, a_{t-1})$.
        obs_model: ObservationModel
            The observation model, $g(z_t \mid s_t)$.
        policy_prior: RecurrentPolicy
            The stochastic policy, $\pi_\phi$.
        policy_prior_params: Parameters
            Parameters of recurrent policy $\phi$.
        reward_fn: RewardFn
            The reward function, $r(s_t, a_t)$.
        slew_rate_penalty: float
            The slew rate penalty.
        tempering: float
            The tempering parameter, $\eta$.
        reference: Reference
            Reference trajectory of the conditional particle filter.
        history_state: HistoryState
            Leaves have shape (N, ...).
        belief_state: BeliefState
            Leaves have shape (N, M, ...).
    """
    num_particles = history_state.weights.shape[0]

    # 1. Resample the history particles.
    key, sub_key = random.split(rng_key)
    history_state = resample_history(
        sub_key, history_state, multinomial_resampling, conditional=True
    )
    particles = history_state.particles
    resampling_idx = history_state.resampling_indices
    belief_state = jax.tree.map(lambda x: x[resampling_idx], belief_state)

    # 2. Resample the belief particles.
    key, sub_keys = custom_split(key, num_particles + 1)
    belief_state = jax.vmap(resample_belief, in_axes=(0, 0, None))(
        sub_keys, belief_state, systematic_resampling
    )

    # 3. Sample new actions.
    key, sub_key = random.split(key)
    carry, actions, _, _ = policy_prior.sample_and_log_prob(
        sub_key, particles.carry, particles.actions, particles.observations, policy_prior_params
    )

    # replace zeroth carry and action with reference carry and action
    carry = jax.tree.map(lambda x, y: x.at[0].set(y), carry, reference.history_particles.carry)
    actions = actions.at[0].set(reference.history_particles.actions)

    # 4. Propagate the belief particles.
    key, sub_keys = custom_split(key, num_particles + 1)
    belief_particles = jax.vmap(propagate_belief, in_axes=(0, None, 0, 0))(
        sub_keys, trans_model, belief_state.particles, actions
    )
    belief_state = belief_state._replace(particles=belief_particles)

    # replace zeroth belief state with reference belief state
    belief_state = jax.tree.map(
        lambda x, y: x.at[0].set(y), belief_state, reference.belief_state
    )

    # 5. Sample new observations.
    key, sub_keys = custom_split(key, num_particles + 1)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        sub_keys, obs_model, belief_state
    )

    # replace zeroth observations with reference observation
    observations = observations.at[0].set(reference.history_particles.observations)

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
        "init_prior",
        "policy_prior",
        "trans_model",
        "obs_model",
        "reward_fn"
    )
)
def csmc(
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
    reference: Reference,
) -> tuple[HistoryState, BeliefState, BeliefInfo, Array]:
    """
    Perform the Conditional Sequential Monte Carlo (CSMC) algorithm.

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
        obs_model: ObservationModel
            The observation model.
        policy_prior: RecurrentPolicy
            The recurrent policy.
        policy_prior_params: Parameters
            Parameters of the recurrent policy.
        reward_fn: RewardFn
            The reward function.
        slew_rate_penalty: float
            The slew rate penalty.
        tempering: float
            The tempering parameter.
        reference: Reference
            Reference trajectory of the conditional particle filter.

    Returns:
        tuple[HistoryState, BeliefState, Array]
            All history and belief states after running the SMC algorithm along
            with the normalizing constant estimate.
    """

    def csmc_loop(carry, args):
        history_state, belief_state, log_marginal = carry
        time_idx, key, ref_state = args

        history_state, belief_state, belief_info, log_marginal_incr = \
            csmc_step(
                rng_key=key,
                policy_prior=policy_prior,
                policy_prior_params=policy_prior_params,
                trans_model=trans_model,
                obs_model=obs_model,
                reward_fn=reward_fn,
                slew_rate_penalty=slew_rate_penalty,
                tempering=tempering,
                reference=ref_state,
                history_state=history_state,
                belief_state=belief_state,
                time_idx=time_idx,
            )

        log_marginal += log_marginal_incr
        return (history_state, belief_state, log_marginal), \
            (history_state, belief_state, belief_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_history_state, init_belief_state, init_belief_info = csmc_init(
        rng_key=init_key,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=init_prior,
        policy_prior=policy_prior,
        obs_model=obs_model,
        reference=jax.tree.map(lambda x: x[0], reference),
    )

    (_, _, log_marginal), (history_states, belief_states, belief_infos) = jax.lax.scan(
        f=csmc_loop,
        init=(init_history_state, init_belief_state, jnp.array(0.0)),
        xs=(
            jnp.arange(1, num_time_steps + 1),  # time indices
            random.split(loop_key, num_time_steps),  # random keys
            jax.tree.map(lambda x: x[1:], reference)  # reference trajectory
        ),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    history_states = concat_trees(init_history_state, history_states)
    belief_states = concat_trees(init_belief_state, belief_states)
    belief_infos = concat_trees(init_belief_info, belief_infos)
    return history_states, belief_states, belief_infos, log_marginal
