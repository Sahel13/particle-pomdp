from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp
from chex import Numeric, PRNGKey
from distrax import Chain, Distribution
from flax.core import FrozenDict
from jax import Array

from ppomdp.core import RewardFn, TransitionModel
from ppomdp.policy import squash_policy


class Environment(NamedTuple):
    state_dim: int
    action_dim: int
    num_time_steps: int
    action_scale: Numeric
    action_shift: Numeric
    trans_model: TransitionModel
    reward_fn: RewardFn
    prior_dist: Distribution


class OuterState(NamedTuple):
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    time_steps: Array
    dones: Array
    episodic_rewards: Array


class SoftQNetwork(nn.Module):
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, state: Array, action: Array) -> Array:
        y = jnp.concatenate([state, action], axis=-1)
        for size in self.hidden_sizes:
            y = nn.relu(nn.Dense(size)(y))
        return nn.Dense(1)(y)


class ActorNetwork(nn.Module):
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_max: float = 2
    log_std_min: float = -5

    @nn.compact
    def __call__(self, state: Array) -> tuple[Array, Array]:
        y = state
        for size in self.hidden_sizes:
            y = nn.relu(nn.Dense(size)(y))

        mean = nn.Dense(self.action_dim)(y)
        log_std = nn.tanh(nn.Dense(self.action_dim)(y))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        return mean, log_std


def sample_and_log_prob(
    rng_key: PRNGKey,
    states: Array,
    params: FrozenDict,
    network: ActorNetwork,
    bijector: Chain,
) -> tuple[Array, Array, Array]:
    """Sample actions and compute log probabilities."""
    mean, log_std = network.apply({"params": params}, states)
    dist = squash_policy(mean, log_std, bijector)
    sampled_action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    mean_action = bijector.forward(mean)
    return sampled_action, log_prob, mean_action
