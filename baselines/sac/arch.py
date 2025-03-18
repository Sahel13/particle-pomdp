from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.arch import MLPDecoder


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    layer_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, state: Array, time_idx: Array) -> [Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, time_idx[..., None]], -1)
        for size in self.layer_sizes:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.output_dim)(x), log_std


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    layer_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(self, state: Array, action: Array, time_idx: Array):
        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, action, time_idx[..., None]], -1)
        values = [MLPDecoder(self.layer_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
