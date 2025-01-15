from collections.abc import Callable
from typing import NamedTuple

import flax.linen as nn
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Chain, ScalarAffine
from flax.core import FrozenDict
from jax import Array
from jax.typing import ArrayLike

from ppomdp.bijector import Tanh
from ppomdp.policy import squash_policy


class OuterState(NamedTuple):
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    time_steps: Array
    dones: Array
    episodic_rewards: Array


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, data: Array):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(hidden_size)(hidden)
            if i != len(self.layer_sizes) - 1:
                hidden = nn.relu(hidden)
        return hidden


class QNetworks(nn.Module):
    obs_fn: Callable[[Array], Array]
    n_critics: int = 2
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, states: Array, actions: Array):
        observations = self.obs_fn(states)
        hidden = jnp.concatenate([observations, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_sizes + (1,),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class ActorNetwork(nn.Module):
    action_dim: int
    init_log_std: Array
    obs_fn: Callable[[Array], Array]
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_max: float = 2
    log_std_min: float = -5

    @nn.compact
    def __call__(self, state: Array) -> tuple[Array, Array]:
        y = self.obs_fn(state)
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
    params: FrozenDict,
    states: Array,
    network: ActorNetwork,
    action_scale: ArrayLike,
    action_shift: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Sample actions and compute their log probabilities."""
    mean, log_std = network.apply(params, states)
    bijector = Chain([ScalarAffine(action_scale, action_shift), Tanh()])
    dist = squash_policy(mean, log_std, bijector)
    sampled_action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    mean_action = bijector.forward(mean)
    return sampled_action, log_prob, mean_action
