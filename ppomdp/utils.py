from typing import Callable, Union

import math
from functools import partial

import jax
from jax import Array, random
from jax import numpy as jnp
from distrax import Distribution

from ppomdp.core import (
    PRNGKey,
    Parameters,
    BeliefState,
    RecurrentPolicy,
    AttentionPolicy,
    LinearPolicy,
    TransitionModel,
    ObservationModel,
)


@partial(jax.jit, static_argnames=("data_size", "batch_size", "skip_last"))
def batch_data(
    rng_key: Array,
    data_size: int,
    batch_size: int,
    skip_last: bool = True,
) -> list[Array]:
    """
    Generates batches of data indices by shuffling the data indices randomly and splitting them into
    subsets of the specified batch size. Optionally, incomplete batches can be included or excluded.

    Args:
        rng_key (Array): Random key for deterministic shuffling of the data indices.
        data_size (int): Total number of data points to batch.
        batch_size (int): Size of each batch.
        skip_last (bool, optional): Whether to exclude the last incomplete batch when the data
            size is not perfectly divisible by the batch size. Defaults to True.

    Returns:
        list[Array]: A list of arrays, where each array contains indices corresponding to a batch
            of the specified size.
    """
    batch_idx = random.permutation(rng_key, data_size)

    if skip_last:
        # Skip incomplete batch
        num_batches = data_size // batch_size
        batch_idx = batch_idx[: num_batches * batch_size]
    else:
        # Include incomplete batch
        num_batches = math.ceil(data_size / batch_size)

    batch_idx = jnp.array_split(batch_idx, num_batches)
    return batch_idx


def stitch_arrays(a: Array, b: Array, stitch: int, max_size: int):
    """
    Stitches two input arrays element-wise based on a stitching boundary value
    and a maximum size constraint.

    The function determines elements of the output array based on their position
    relative to the defined stitching boundary. Elements from the first input array
    are selected for positions less than or equal to the stitching boundary, while
    elements from the second array are selected for positions beyond the stitching
    boundary, up to the specified maximum size.

    Args:
        a (Array): The first input array from which elements are selected up to the
            stitching boundary.
        b (Array): The second input array from which elements are selected beyond
            the stitching boundary.
        stitch (int): The index position defining the stitching boundary between the
            two arrays.
        max_size (int): The maximum size of the resulting output array.

    Returns:
        Array: The stitched array comprising elements from the two input arrays
            based on the stitching boundary and constrained by the maximum size.
    """

    return jnp.where(jnp.arange(max_size)[:, None] <= stitch, a, b)


def custom_split(rng_key: PRNGKey, num: int):
    r"""Splits a random number generator key into multiple sub-keys.

    Args:
        rng_key: PRNGKey
            The random number generator key to split
        num: int
            The number of sub-keys to generate

    Returns:
        tuple:
            A tuple containing the next key and an array of sub-keys
    """
    key, *sub_keys = random.split(rng_key, num)
    return key, jnp.array(sub_keys)


@jax.jit
def prepare_trajectories(key: Array, actions: Array, beliefs: BeliefState):
    num_time_steps, num_history_particles, _ = actions.shape
    _, _, num_particles, _ = beliefs.particles.shape

    from ppomdp.smc.utils import resample_belief, systematic_resampling
    def _resample_belief(key, belief):
        key, sub_keys = custom_split(key, num_history_particles + 1)
        return jax.vmap(resample_belief, in_axes=(0, 0, None))(
            sub_keys, belief, systematic_resampling
        )

    key, belief_keys = custom_split(key, num_time_steps + 1)
    resampled_beliefs = jax.vmap(_resample_belief, in_axes=(0, 0))(
        belief_keys, beliefs
    )

    particles, weights = resampled_beliefs.particles, resampled_beliefs.weights
    actions = actions[1:].reshape((-1, actions.shape[-1]))
    particles = particles[:-1].reshape((-1, num_particles, particles.shape[-1]))
    weights = weights[:-1].reshape((-1, weights.shape[-1]))
    return actions, particles, weights


@partial(
    jax.jit,
    static_argnames=(
        "num_time_steps",
        "num_trajectory_samples",
        "num_belief_particles",
        "policy",
        "init_dist",
        "belief_prior",
        "trans_model",
        "obs_model",
        "reward_fn",
    )
)
def policy_evaluation(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_trajectory_samples: int,
    num_belief_particles: int,
    init_dist: Distribution,
    belief_prior: Distribution,
    policy: Union[RecurrentPolicy, AttentionPolicy, LinearPolicy],
    policy_params: Parameters,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: Callable,
    stochastic: bool = True,
):
    """Evaluates a policy in a partially observable Markov decision process (POMDP)
    environment over a specified number of samples and time steps.

    This function computes the average cumulative reward achieved by a policy in a
    given POMDP environment. It uses JAX to enable high-performance automatic
    differentiation and parallel computation. Actions, states, and observations are
    sampled sequentially in a scan loop, and the rewards are accumulated to evaluate
    the policy's performance.

    Args:
        rng_key (PRNGKey): Random number generator key, used to ensure reproducibility
            in sampling actions, states, and observations.
        num_time_steps (int): Number of time steps to evaluate the policy.
        num_trajectory_samples (int): Number of sampled trajectories to evaluate the policy.
        num_belief_particles (int): Number of belief particles to use for the SMC algorithm.
        init_dist (Distribution): Initial distribution of the states.
        belief_prior (Distribution): Prior distribution of the beliefs.
        policy (RecurrentPolicy): The recurrent policy to evaluate. It provides
            methods for sampling actions and resetting its internal state.
        policy_params (Parameters): Policy parameters used during action sampling.
        trans_model (TransitionModel): Transition model of the POMDP environment.
        obs_model (ObservationModel): Observation model of the POMDP environment.
        reward_fn (Callable): Reward function of the POMDP environment.
        stochastic (bool): Specifies whether the sampled actions or deterministic
            mean actions are used during evaluation. Defaults to True.

    Returns:
        Tuple[float, jnp.ndarray, jnp.ndarray, jnp.ndarray]: A tuple containing:
            - The average cumulative reward computed over the sampled trajectories.
            - A tensor of sampled states for all time steps and trajectories.
            - A tensor of sampled actions for all time steps and trajectories.
            - A tensor of sampled beliefs for all time steps and trajectories.
    """

    from ppomdp.smc.utils import (
        initialize_belief, update_belief, resample_belief, systematic_resampling
    )

    def body(val, key):
        states, actions, carry, observations, beliefs, time_idx = val

        # Sample actions.
        key, action_key = random.split(key)
        if isinstance(policy, RecurrentPolicy):
            carry, samples, means = policy.sample(
                rng_key=action_key,
                carry=carry,
                actions=actions,
                observations=observations,
                params=policy_params
            )
        else:  # isinstance(AttentionPolicy)
            key, belief_keys = custom_split(key, num_trajectory_samples + 1)
            resampled_beliefs = jax.vmap(resample_belief, in_axes=(0, 0, None))(
                belief_keys, beliefs, systematic_resampling
            )

            carry = None
            samples, means = policy.sample(
                rng_key=action_key,
                particles=resampled_beliefs.particles,
                weights=resampled_beliefs.weights,
                params=policy_params
            )

        actions = jax.lax.select(stochastic, samples, means)

        # Sample next states.
        key, state_keys = custom_split(key, num_trajectory_samples + 1)
        states = jax.vmap(trans_model.sample)(state_keys, states, actions)

        # Compute rewards.
        rewards = jax.vmap(reward_fn, (0, 0, None))(states, actions, time_idx)

        # Sample observations.
        obs_keys = random.split(key, num_trajectory_samples)
        observations = jax.vmap(obs_model.sample)(obs_keys, states)

        # Update beliefs
        belief_keys = random.split(key, num_trajectory_samples)
        beliefs = jax.vmap(update_belief, (0, None, None, 0, 0, 0))(
            belief_keys, trans_model, obs_model, beliefs, observations, actions
        )

        return (states,  actions, carry, observations, beliefs, time_idx + 1), \
            (states, actions, beliefs, rewards)

    key, state_key = random.split(rng_key)
    init_states = init_dist.sample(seed=state_key, sample_shape=num_trajectory_samples)

    key, obs_keys = custom_split(key, num_trajectory_samples + 1)
    init_observations = jax.vmap(obs_model.sample)(obs_keys, init_states)

    init_actions = jnp.zeros((num_trajectory_samples, policy.dim))
    init_carry = policy.reset(num_trajectory_samples)

    key, belief_keys = custom_split(key, num_trajectory_samples + 1)
    init_beliefs = jax.vmap(initialize_belief, in_axes=(0, None, None, 0, None))(
        belief_keys, belief_prior, obs_model, init_observations, num_belief_particles
    )

    _, (states, actions, beliefs, rewards) = jax.lax.scan(
        f=body,
        init=(init_states, init_actions, init_carry, init_observations, init_beliefs, 1),
        xs=random.split(key, num_time_steps),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    states = concat_trees(init_states,  states)
    beliefs = concat_trees(init_beliefs, beliefs)
    return rewards, states, actions, beliefs


def damping_schedule(
    step: int,
    total_steps: int,
    init_value: float = 0.1,
    max_value: float = 0.95,
    steepness: float = 1.0,
) -> float:
    r"""Generate smooth damping factor that increases over training.

    Uses a sigmoid schedule to gradually increase damping from init_value to max_value:
    $\beta = \frac{1}{1 + e^{-\alpha x}}$
    where x is a scaled time variable and α is the steepness parameter.

    Args:
        step: int
            Current training step
        total_steps: int
            Total number of training steps
        steepness: float, optional
            Controls slope of sigmoid curve. Defaults to 1.0
        init_value: float, optional
            Initial damping value. Defaults to 0.1
        max_value: float, optional
            Maximum damping value. Defaults to 0.95

    Returns:
        float:
            Damping factor between init_value and max_value
    """
    # Scale step to range roughly [-6, 6] for sigmoid input
    # This makes the steepest change happen around the middle of training
    x = (step / total_steps) * 12.0 - 6.0

    # Calculate sigmoid
    beta = 1 / (1 + jnp.exp(-steepness * x))

    # Rescale beta from [0, 1] range to [init_value, max_value] range
    beta_scaled = init_value + (max_value - init_value) * beta

    # Ensure value does not exceed max_value (due to potential float issues)
    # Note: If init_value < max_value, beta is <= 1, this might seem redundant,
    # but adds robustness.
    return jnp.minimum(beta_scaled, max_value)


def weighted_huber_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray, weights: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Computes the weighted Huber loss element-wise and sums the result.

    The Huber loss is defined as:
    L_delta(a) = { 0.5 * a^2                   if |a| <= delta
               { delta * (|a| - 0.5 * delta)  if |a| > delta
    where a = y_pred - y_true.

    The weighted Huber loss is then sum(weights * L_delta(y_pred - y_true)).

    Args:
    y_pred: Predicted values (JAX array).
    y_true: True values (JAX array, must have the same shape as y_pred).
    weights: Weighting matrix/array (JAX array, must have the same shape as y_pred).
            Each element defines the importance of the corresponding element-wise loss.
    delta: The threshold parameter for the Huber loss. Determines the point where
           the loss transitions from quadratic to linear. Defaults to 1.0.

    Returns:
    A scalar JAX array representing the total weighted Huber loss.

    Raises:
      ValueError: If input arrays do not have compatible shapes for element-wise operations.
                  (JAX will typically raise its own errors during computation).
    """
    # Ensure inputs have compatible shapes (JAX handles broadcasting, but explicit checks can be added)
    # Note: JAX usually raises errors if shapes are incompatible during operations.
    # Example explicit check (optional):
    # if y_pred.shape != y_true.shape or y_pred.shape != weights.shape:
    #   raise ValueError(f"Shapes must match: y_pred={y_pred.shape}, "
    #                    f"y_true={y_true.shape}, weights={weights.shape}")

    error = y_pred - y_true
    abs_error = jnp.abs(error)

    # Quadratic loss part (for |error| <= delta)
    quadratic_loss = 0.5 * jnp.square(error)

    # Linear loss part (for |error| > delta)
    linear_loss = delta * (abs_error - 0.5 * delta)

    # Combine using jnp.where based on the condition |error| <= delta
    elementwise_huber = jnp.where(abs_error <= delta, quadratic_loss, linear_loss)

    # Apply weights element-wise
    weighted_elementwise_loss = weights * elementwise_huber

    # Sum the weighted losses
    total_loss = jnp.sum(weighted_elementwise_loss)

    return total_loss