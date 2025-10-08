from typing import Callable

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from ppomdp.policy.arch import DualHeadMLPDecoder, MLPDecoder

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    encoding_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1

    @nn.compact
    def __call__(self, particles: Array) -> tuple[Array, Array]:
        # Prepare the input.
        mean_particles = jnp.mean(particles, axis=-2, keepdims=True)
        x = jnp.concatenate([particles, mean_particles], -2)
        x = self.feature_fn(x)
        x = nn.DenseGeneral(features=self.encoding_dim, axis=(-2, -1))(x)

        # Get the action.
        mean, log_std = DualHeadMLPDecoder(self.hidden_sizes, self.output_dim)(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (nn.tanh(log_std) + 1)
        return mean, log_std


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
