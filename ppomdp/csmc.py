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
    systematic_resampling
)


def csmc_init(
    rng_key: PRNGKey,
    num_history_particles: int,
    num_belief_particles: int,
    prior_dist: Distribution,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reference: Reference,
) -> tuple[HistoryState, BeliefState, BeliefInfo]:
    r"""Initialize the history and belief states for the nested CSMC algorithm.

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
        reference: Reference
            Reference state of the conditional particle filter.
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
    init_carry = policy.reset(num_history_particles)
    dummy_actions = jnp.zeros((num_history_particles, policy.dim))
    dummy_log_probs = jnp.zeros(num_history_particles)

    # replace zeroth carry, action, and log_prob with reference carry, action, and log_prob
    init_carry = jax.tree_map(lambda x, y: x.at[0].set(y), init_carry, reference.history_particles.carry)
    dummy_actions = dummy_actions.at[0].set(reference.history_particles.actions)
    dummy_log_probs = dummy_log_probs.at[0].set(reference.history_particles.log_probs)

    history_particles = HistoryParticles(observations, dummy_actions, init_carry, dummy_log_probs)
    history_state = HistoryState(
        particles=history_particles,
        log_weights=jnp.zeros(num_history_particles),
        weights=jnp.ones(num_history_particles) / num_history_particles,
        resampling_indices=jnp.zeros(num_history_particles, dtype=jnp.int32),
        rewards=jnp.zeros(num_history_particles),
    )
    return history_state, belief_state, belief_info


def csmc_step(
    time_idx: int,
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    policy: RecurrentPolicy,
    params: Parameters,
    reward_fn: RewardFn,
    tempering: float,
    slew_rate_penalty: float,
    reference: Reference,
    history_state: HistoryState,
    belief_state: BeliefState,
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
    keys = random.split(key, num_particles + 1)
    belief_state = jax.vmap(resample_belief, in_axes=(0, 0, None))(
        keys[1:], belief_state, multinomial_resampling
    )

    # 3. Sample new actions.
    key, sub_key = random.split(keys[0])
    carry, actions, log_probs = policy.sample_and_log_prob(
        sub_key, particles.carry, particles.observations, params
    )

    # replace zeroth carry, action, and log_prob with reference carry, action, and log_prob
    carry = jax.tree.map(lambda x, y: x.at[0].set(y), carry, reference.history_particles.carry)
    actions = actions.at[0].set(reference.history_particles.actions)
    log_probs = log_probs.at[0].set(reference.history_particles.log_probs)

    # 4. Propagate the belief particles.
    keys = random.split(key, num_particles + 1)
    belief_particles = jax.vmap(propagate_belief, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, belief_state.particles, actions
    )
    belief_state = belief_state._replace(particles=belief_particles)

    # replace zeroth belief state with reference belief state
    belief_state = jax.tree.map(
        lambda x, y: x.at[0].set(y), belief_state, reference.belief_state
    )

    # 5. Sample new observations.
    keys = random.split(keys[0], num_particles)
    observations = jax.vmap(sample_marginal_obs, in_axes=(0, None, 0))(
        keys, obs_model, belief_state
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
def csmc(
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

        history_state, belief_state, belief_info, log_marginal_incr = csmc_step(
            time_idx,
            key,
            trans_model,
            obs_model,
            policy,
            params,
            reward_fn,
            tempering,
            slew_rate_penalty,
            ref_state,
            history_state,
            belief_state,
        )

        log_marginal += log_marginal_incr
        return (history_state, belief_state, log_marginal), (history_state, belief_state, belief_info)

    init_key, loop_key = random.split(rng_key, 2)
    init_history_state, init_belief_state, init_belief_info = csmc_init(
        init_key,
        num_history_particles,
        num_belief_particles,
        prior_dist,
        obs_model,
        policy,
        params,
        jax.tree.map(lambda x: x[0], reference),
    )

    (_, _, log_marginal), (history_states, belief_states, belief_infos) = jax.lax.scan(
        csmc_loop,
        (init_history_state, init_belief_state, jnp.array(0.0)),
        (
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
