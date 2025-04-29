from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.policy.arch import MLPDecoder


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    recurr_size: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, particles: Array) -> [Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        # Prepare the input.
        mean_particles = jnp.mean(particles, axis=-2, keepdims=True)
        x = jnp.concatenate([particles, mean_particles], -2)
        x = self.feature_fn(x)

        # Get the action.
        x = nn.DenseGeneral(features=self.hidden_sizes[0], axis=(-2, -1))(x)
        x = nn.relu(x)
        return MLPDecoder(self.hidden_sizes[1:], self.output_dim)(x), log_std


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, states: Array, action: Array, time_idx: Array):
        features = self.feature_fn(states)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([features, action, time_idx[..., None]], -1)
        return MLPDecoder(self.hidden_sizes, 1)(x)
