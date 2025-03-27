from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.arch import MLPDecoder


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    recurr_size: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, particles: Array, weights: Array) -> [Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        features = self.feature_fn(particles)
        inputs = jnp.concatenate([features, weights[..., None]], -1)
        encoding = nn.RNN(nn.GRUCell(self.recurr_size))(inputs)
        encoding = jnp.take(encoding, -1, axis=-2)
        return MLPDecoder(self.hidden_sizes, self.output_dim)(encoding), log_std


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
