from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp, Array, random
from distrax import Distribution

from ppomdp.core import (
    PRNGKey,
    Parameters,
    Carry,
    HistoryState,
    BeliefState,
    TransitionModel,
    ObservationModel,
    RewardFn,
    RecurrentPolicy
)

ESS_THRESHOLD = 0.75


def resample_belief(
    rng_key: PRNGKey,
    belief_state: BeliefState,
    resample_fn: Callable,
) -> BeliefState:
    r"""Resample the belief particles for a single history trajectory using adaptive resampling.

    This function implements adaptive resampling - it only triggers resampling when
    the effective sample size (ESS) falls below 75% of the total number of particles.
    This helps prevent particle degeneracy while avoiding unnecessary resampling.

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling operations
        belief_state: BeliefState
            Current belief state containing:
            - particles: State particles with shape (M, ...)
            - weights: Normalized importance weights
            - log_weights: Log of importance weights
            - resampling_indices: Indices from previous resampling
        resample_fn: Callable
            Function implementing the resampling strategy (e.g. systematic, multinomial)

    Returns:
        BeliefState:
            Updated belief state after resampling (if triggered) with:
            - Resampled particles
            - Reset weights to uniform
            - Recorded resampling indices

    Notes:
        - If ESS > 0.75M, returns original state with only indices updated
        - If ESS ≤ 0.75M, performs resampling and resets weights
        - The effective sample size is calculated as:
          $ESS = \frac{(\sum_{m=1}^M w_m)^2}{\sum_{m=1}^M w_m^2}$
    """
    num_particles = belief_state.particles.shape[0]

    def true_fn(state: BeliefState) -> BeliefState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        return BeliefState(
            particles=state.particles[resampling_idx],
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
        )

    def false_fn(state: BeliefState) -> BeliefState:
        resampling_idx = jnp.arange(num_particles, dtype=jnp.int32)
        return state._replace(resampling_indices=resampling_idx)

    resampled_state = jax.lax.cond(
        effective_sample_size(belief_state.log_weights) < ESS_THRESHOLD * num_particles,
        true_fn,
        false_fn,
        belief_state,
    )
    return resampled_state


def propagate_belief(
    rng_key: PRNGKey,
    model: TransitionModel,
    particles: Array,
    action: Array,
) -> Array:
    r"""Propagate belief particles forward using the transition model.

    This function advances each particle through the state transition model
    to get the next state distribution. It handles vectorized operations across
    all particles efficiently.

    Args:
        rng_key: PRNGKey
            Random number generator key for sampling
        model: TransitionModel
            The transition model for the state.
        particles: Array
            The state particles $\{s_{t-1}^{nm}\}_{m=1}^M$ associated with
            the n-th history trajectory. Has shape (M, ...).
        action: Array
            The action $a_{t-1}^n$.

    Returns:
        Array:
            The propagated particles $\{s_t^{nm}\}_{m=1}^M$ with shape (M, ...).
    """
    num_particles = particles.shape[0]
    rng_keys = random.split(rng_key, num_particles)
    return jax.vmap(model.sample, in_axes=(0, 0, None))(rng_keys, particles, action)


def reweight_belief(
    model: ObservationModel,
    state: BeliefState,
    observation: Array
) -> BeliefState:
    r"""Update particle weights based on new observation.

    Computes importance weights for particles by evaluating the observation
    likelihood. Weights are normalized to sum to 1.

    Args:
        model: ObservationModel
            The observation model.
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).
        observation: Array
            The observation $z_t^n$.

    Returns:
        BeliefState:
            Updated belief state with new weights computed as:
            $w_t^{nm} = \frac{w_{t-1}^{nm} p(z_t^n | s_t^{nm})}{\sum_{m'=1}^M w_{t-1}^{nm'} p(z_t^n | s_t^{nm'})}$
    """
    log_weights = jax.vmap(model.log_prob, in_axes=(None, 0))(observation, state.particles)
    log_weights += state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    return state._replace(log_weights=log_weights, weights=weights)


def sample_marginal_obs(
    rng_key: PRNGKey,
    obs_model: ObservationModel,
    state: BeliefState
) -> Array:
    r"""Sample an observation from the marginal observation distribution.

    Samples an observation from the weighted mixture of observation models:
    $z_t^n \sim \sum_{m=1}^M W_{s,t}^{nm} h(z_t \mid s_t^{nm})$

    Args:
        rng_key: PRNGKey
            Random number generator key
        obs_model: ObservationModel
            The observation model
        state: BeliefState
            The belief state associated with the n-th history trajectory.
            Leaves have shape (M, ...).

    Returns:
        Array:
            A sampled observation from the marginal distribution
    """
    key, sub_key = random.split(rng_key)
    x = random.choice(sub_key, state.particles, p=state.weights)
    return obs_model.sample(key, x)


def expected_reward(
    belief_state: BeliefState,
    action: Array,
    time_idx: int,
    reward_fn: RewardFn
) -> Array:
    r"""Calculate the expected reward for a given particle and belief state.

    Computes the expected reward by taking a weighted average over the belief state:
    $\mathbb{E}[r_t] = \sum_{m=1}^M w_t^{nm} r(s_t^{nm}, a_t^n)$

    Args:
        belief_state: BeliefState
            The belief state containing particles and weights
        action: Array
            Action particle
        time_idx: int
            The current time index
        reward_fn: RewardFn
            The reward function, r(s_{t}, a_{t-1})

    Returns:
        Array:
            The expected reward for the given particle and belief state
    """
    rewards = jax.vmap(reward_fn, in_axes=(0, None, None))(
        belief_state.particles, action, time_idx
    )
    if rewards.ndim != 1:
        raise ValueError("The reward function must return a scalar value.")
    return jnp.sum(rewards * belief_state.weights)


def log_potential(
    belief_state: BeliefState,
    action: Array,
    prev_action: Array,
    time_idx: int,
    reward_fn: RewardFn,
    slew_rate_penalty: float,
    tempering: float,
) -> tuple[Array, Array]:
    r"""Estimate the log potential function.

    The estimate for the log potential function is given by:
    $\log g_t^n = \eta \left( \sum_{m=1}^M W_{s,t}^{nm} r_t(s_t^{nm}, a_t^n) - \lambda \|a_t^n - a_{t-1}^n\|^2 \right)$

    Args:
        belief_state: BeliefState
            The belief state associated with the n-th history trajectory
        action: Array
            The current action
        prev_action: Array
            The previous action
        time_idx: int
            The current time index
        reward_fn: RewardFn
            The reward function
        slew_rate_penalty: float
            The penalty coefficient for action changes
        tempering: float
            The tempering parameter η

    Returns:
        tuple[Array, Array]:
            - The tempered log potential value
            - The raw reward value before tempering
    """
    rewards = expected_reward(belief_state, action, time_idx, reward_fn)
    mod_rewards = rewards - slew_rate_penalty * jnp.dot(
        action - prev_action, action - prev_action
    )
    return tempering * mod_rewards, rewards


def resample_history(
    rng_key: PRNGKey,
    history_state: HistoryState,
    resample_fn: Callable,
    conditional: bool = False,
) -> HistoryState:
    r"""Resample history particles using adaptive resampling.

    Performs adaptive resampling on the history state, similar to resample_belief,
    but operates on the full history trajectory. Can optionally perform conditional
    resampling where the first particle is preserved.

    Args:
        rng_key: PRNGKey
            Random number generator key
        history_state: HistoryState
            Current history state containing particles, weights, and rewards
        resample_fn: Callable
            Function implementing the resampling strategy
        conditional: bool, optional
            If True, preserves the first particle during resampling

    Returns:
        HistoryState:
            Resampled history state with updated particles and uniform weights
    """
    num_particles = history_state.weights.shape[0]

    def true_fn(state: HistoryState) -> HistoryState:
        resampling_idx = resample_fn(rng_key, state.weights, num_particles)
        # Set zeroth resampling index to zero if conditional resampling is enabled
        resampling_idx = jax.lax.select(
            conditional, resampling_idx.at[0].set(0), resampling_idx
        )
        resampled_particles = jax.tree.map(lambda x: x[resampling_idx], state.particles)
        resampled_rewards = state.rewards[resampling_idx]
        return HistoryState(
            particles=resampled_particles,
            log_weights=jnp.zeros(num_particles),
            weights=jnp.ones(num_particles) / num_particles,
            resampling_indices=resampling_idx,
            rewards=resampled_rewards,
        )

    def false_fn(state: HistoryState) -> HistoryState:
        resampling_idx = jnp.arange(num_particles, dtype=jnp.int32)
        return state._replace(resampling_indices=resampling_idx)

    predicate = effective_sample_size(history_state.log_weights) < ESS_THRESHOLD * num_particles
    resampled_state = jax.lax.cond(predicate, true_fn, false_fn, history_state)
    return resampled_state


def systematic_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    r"""Perform systematic resampling of particles based on their weights.

    Implements systematic resampling which provides lower variance than multinomial
    resampling. The algorithm generates a single uniform random number and uses it
    to create a deterministic grid of points for resampling.

    Args:
        rng_key: PRNGKey
            Random number generator key
        weights: Array
            The weights of the particles
        num_samples: int
            The number of samples to draw

    Returns:
        Array:
            The indices of the resampled particles

    Notes:
        The systematic resampling algorithm:
        1. Generate a single uniform random number u ~ U[0,1]
        2. Create a grid of points: u_i = (i + u)/N for i = 0,...,N-1
        3. For each u_i, find the particle whose cumulative weight interval contains u_i
    """
    n = weights.shape[0]
    u = random.uniform(rng_key, ())
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1).astype(jnp.int32)


def multinomial_resampling(rng_key: PRNGKey, weights: Array, num_samples: int) -> Array:
    r"""Perform multinomial resampling of particles based on their weights.

    Implements multinomial resampling where each particle is sampled independently
    according to its weight. This is simpler but has higher variance than systematic
    resampling.

    Args:
        rng_key: PRNGKey
            Random number generator key
        weights: Array
            The weights of the particles
        num_samples: int
            The number of samples to draw

    Returns:
        Array:
            The indices of the resampled particles

    Notes:
        The multinomial resampling algorithm:
        1. Generate N independent uniform random numbers
        2. For each random number, find the particle whose cumulative weight interval contains it
    """
    idx = random.choice(rng_key, num_samples, shape=(num_samples,), p=weights)
    return idx.astype(jnp.int32)


def observation_marginal_log_prob(
    states: Array,
    observation: Array,
    obs_model: ObservationModel,
):
    r"""Compute the log marginal likelihood of a single observation.

    Calculates the log probability of an observation by marginalizing over the state particles:
    $\log p(z_t^n) = \log \left( \frac{1}{M} \sum_{m=1}^M p(z_t^n | s_t^{nm}) \right)$

    Args:
        states: Array
            State particles
        observation: Array
            The observation
        obs_model: ObservationModel
            The observation model

    Returns:
        Array:
            The log marginal likelihood of the observation
    """
    num_particles = states.shape[0]
    log_probs = jax.vmap(obs_model.log_prob, in_axes=(None, 0))(observation, states)
    return jax.nn.logsumexp(log_probs) - jnp.log(num_particles)


def action_sequence_log_prob(
    policy: RecurrentPolicy,
    policy_params: Parameters,
    actions_sequence: Array,
    observations_sequence: Array,
    init_carry: list[Carry],
    init_action: Array,
    init_observation: Array,
    start_time_idx: int,
) -> Array:
    r"""Compute the log probability of a sequence of actions under the policy.

    Calculates $\log p(a_{t:T} | z_{t-1:T-1}) = \sum_{k=t}^T \log p(a_k | h_{k-1}, z_{k-1})$,
    where $h_{k-1}$ represents the policy's internal state (carry) after processing $a_{k-1}$ and $z_{k-1}$.

    Args:
        policy: The recurrent policy to evaluate.
        policy_params: Parameters for the policy.
        actions_sequence: Sequence of actions $a_0, ..., a_T$. Shape (T+1, ...).
        observations_sequence: Sequence of observations $z_0, ..., z_T$. Shape (T+1, ...).
        init_carry: Initial policy carry state before processing $a_t$ and $z_{t-1}$.
        init_action: Initial action to be used with the first observation.
        init_observation: Observation $z_{t-1}$ used for the first action $a_t$.
        start_time_idx: The starting time index $t$.

    Returns:
        The total log probability $\log p(a_{t:T} | z_{t-1:T-1})$.
    """
    num_steps_plus_one, _ = actions_sequence.shape # Length of sequences (T+1)

    def remaining_loop(k, state):
        carry, log_prob = state
        next_carry, log_prob_inc = policy.carry_and_log_prob(
            next_actions=actions_sequence[k],
            carry=carry,
            actions=actions_sequence[k - 1],
            observations=observations_sequence[k - 1],
            params=policy_params,
        )
        return next_carry, log_prob + log_prob_inc

    # get log_prob of first future action
    carry, log_prob = policy.carry_and_log_prob(
        next_actions=actions_sequence[start_time_idx],
        carry=init_carry,
        actions=init_action,
        observations=init_observation,
        params=policy_params,
    )

    # JAX left me no choice: for start_time_idx + 1 => num_steps_plus_one
    # this loop will not execute and log_prob will be the init log_prob.
    # This is intentional behavior!
    _, log_prob = jax.lax.fori_loop(
        lower=start_time_idx + 1,
        upper=num_steps_plus_one,
        body_fun=remaining_loop,
        init_val=(carry, log_prob)
    )
    return log_prob


def transition_marginal_log_prob(
    trans_model: TransitionModel,
    next_belief_state: BeliefState,
    belief_state: BeliefState,
    action: Array,
):
    r"""Compute the sum of log marginal transition probabilities to each next-step particle.

    Calculates the sum over next-step particles $m'$ of the log probability of reaching
    particle $s_{t+1}^{m'}$ from the current belief state $b_t = \{ (s_t^m, w_t^m) \}_{m=1}^M$
    given action $a_t$:
    $\sum_{m'=1}^{M} \log p(s_{t+1}^{m'} | b_t, a_t) = \sum_{m'=1}^{M} \log \left( \sum_{m=1}^{M} w_t^m p(s_{t+1}^{m'} | s_t^m, a_t) \right)$
    Note: This does not represent a standard log probability distribution over the next belief state,
    as it sums log probabilities without incorporating the weights $w_{t+1}^{m'}$ of the next belief state.

    Args:
        trans_model: The state transition model $p(s' | s, a)$.
        next_belief_state: The belief state at the next time step, containing particles $s_{t+1}^{m'}$.
        belief_state: The current belief state $b_t$.
        action: The action $a_t$ taken.

    Returns:
        Array:
            The sum of log probabilities of reaching each next-step particle from the current belief.
    """
    # Inner sum: For a fixed next_particle s_{t+1}^{m'}, compute log sum_m w_t^m p(s_{t+1}^{m'} | s_t^m, a_t)
    def _log_prob_next_particle(next_particle, particles, log_weights, action):
        # Compute log p(next_particle | s_t^m, a_t) for all m
        log_trans_probs = jax.vmap(trans_model.log_prob, in_axes=(None, 0, None))(
            next_particle, particles, action
        )
        # Return log sum_m exp(log_trans_probs_m + log(w_t^m))
        return jax.nn.logsumexp(log_trans_probs + log_weights)

    # Outer sum: Compute the inner sum for all next_particles s_{t+1}^{m'}
    # vmap over the future_belief_state.particles (the s_{t+1}^{m'} samples)
    log_prob_next_particles = jax.vmap(_log_prob_next_particle, in_axes=(0, None, None, None))(
        next_belief_state.particles,
        belief_state.particles,
        belief_state.log_weights,
        action
    )
    return jnp.sum(log_prob_next_particles)


@partial(jnp.vectorize, signature="(m,h),(m)->(h)")
def weighted_mean(particles: Array, weights: Array):
    r"""Compute the weighted mean of the particles.

    Calculates the weighted mean of particles:
    $\mu = \frac{\sum_{m=1}^M w_m x_m}{\sum_{m=1}^M w_m}$

    Args:
        particles: Array
            The particles array of shape (m, h), where m is the number of particles and h is the dimension
        weights: Array
            The weights array of shape (m,)

    Returns:
        Array:
            The weighted mean of the particles of shape (h,)
    """
    return jnp.einsum("mh,m->h", particles, weights) / jnp.sum(weights)


@partial(jnp.vectorize, signature="(m,h),(m)->(h,h)")
def weighted_covar(particles: Array, weights: Array) -> Array:
    r"""Compute the weighted empirical covariance of the particles.

    Calculates the weighted covariance matrix:
    $\Sigma = \frac{\sum_{m=1}^M w_m (x_m - \mu)(x_m - \mu)^T}{\sum_{m=1}^M w_m}$

    Args:
        particles: Array
            The particles array of shape (m, h), where m is the number of particles and h is the dimension
        weights: Array
            The weights array of shape (m,)

    Returns:
        Array:
            The weighted covariance matrix of shape (h, h)
    """
    centered = particles - weighted_mean(particles, weights)
    return jnp.einsum("mh,ml,m->hl", centered, centered, weights) / jnp.sum(weights)


def log_ess(log_weights: Array) -> Array:
    r"""Computes the log of the effective sample size.

    Calculates the log of the effective sample size (ESS):
    $\log(ESS) = 2\log\left(\sum_{m=1}^M w_m\right) - \log\left(\sum_{m=1}^M w_m^2\right)$

    Args:
        log_weights: Array
            Log-weights of particles

    Returns:
        Array:
            The logarithm of the effective sample size
    """
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)


@partial(jnp.vectorize, signature="(m)->()")
def effective_sample_size(log_weights: Array) -> Array:
    r"""Computes the effective sample size.

    Calculates the effective sample size (ESS):
    $ESS = \frac{(\sum_{m=1}^M w_m)^2}{\sum_{m=1}^M w_m^2}$

    Args:
        log_weights: Array
            Log-weights of particles

    Returns:
        Array:
            The effective sample size
    """
    return jnp.exp(log_ess(log_weights))


def initialize_belief(
    rng_key: PRNGKey,
    belief_prior: Distribution,
    obs_model: ObservationModel,
    observation: Array,
    num_belief_particles: int
) -> BeliefState:
    particles = belief_prior.sample(seed=rng_key, sample_shape=(num_belief_particles,))
    log_weights = jax.vmap(obs_model.log_prob, (None, 0))(observation, particles)
    weights = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))
    resampling_indices = jnp.zeros(num_belief_particles, dtype=jnp.int32)
    return BeliefState(
        particles=particles,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_indices
    )


def update_belief(
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    belief_state: BeliefState,
    observation: Array,
    action: Array,
) -> BeliefState:
    key, sub_key = random.split(rng_key, 2)
    resampled_belief = resample_belief(sub_key, belief_state, systematic_resampling)
    key, sub_key = random.split(key, 2)
    particles = propagate_belief(
        rng_key=sub_key,
        model=trans_model,
        particles=resampled_belief.particles,
        action=action
    )
    resampled_belief = resampled_belief._replace(particles=particles)
    return reweight_belief(obs_model, resampled_belief, observation)
