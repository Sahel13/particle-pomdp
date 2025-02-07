from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
from chex import Numeric, PRNGKey
from distrax import Chain, ScalarAffine
from flax.core import FrozenDict
from jax import Array

from ppomdp.bijector import Tanh
from ppomdp.core import InnerState
from ppomdp.policy import GRU, LSTM, Carry, squash_policy


class OuterState(NamedTuple):
    """The joint state for environment simulation."""

    states: Array
    observations: Array
    carry: list[Carry]
    belief_states: InnerState
    actions: Array
    next_states: Array
    next_observations: Array
    next_carry: list[Carry]
    next_belief_states: InnerState
    rewards: Array
    episodic_rewards: Array
    time_steps: Array
    dones: Array


class Transition(NamedTuple):
    """Named tuple to hold training data."""

    states: Array
    observations: Array
    carry: list[Carry]
    actions: Array
    next_states: Array
    next_observations: Array
    next_carry: list[Carry]
    time_steps: Array
    rewards: Array
    dones: Array


@jax.jit
def get_transition(os: OuterState) -> Transition:
    """Get the transition data for training."""

    @jax.vmap
    def belief_state_mean(bs: InnerState) -> Array:
        """Compute the mean of a belief state."""
        return jnp.sum(bs.particles * bs.weights[..., None], axis=0)

    states = belief_state_mean(os.belief_states)
    next_states = belief_state_mean(os.next_belief_states)

    return Transition(
        states=states,
        observations=os.observations,
        carry=os.carry,
        actions=os.actions,
        next_states=next_states,
        next_observations=os.next_observations,
        next_carry=os.next_carry,
        time_steps=os.time_steps,
        rewards=os.rewards,
        dones=os.dones,
    )


def sample_and_log_prob(
    rng_key: PRNGKey,
    params: FrozenDict,
    carry: list[Carry],
    observations: Array,
    network: Union[LSTM, GRU],
    action_scale: Numeric,
    action_shift: Numeric,
) -> tuple[list[Carry], Array, Array, Array]:
    """Sample actions and compute their log probabilities."""
    carry, mean = network.apply({"params": params}, carry, observations)
    bijector = Chain([ScalarAffine(action_scale, action_shift), Tanh()])
    dist = squash_policy(mean, params["log_std"], bijector)
    sampled_action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    mean_action = bijector.forward(mean)
    return carry, sampled_action, log_prob, mean_action
