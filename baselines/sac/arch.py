from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.policy.arch import MLPDecoder


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class PolicyNetwork(nn.Module):
    """SAC Policy Network (Actor)."""
    feature_fn: Callable
    time_norm: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1

    @nn.compact
    def __call__(self, state: Array, time_idx: Array) -> [Array, Array]:
        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, time_idx[..., None]], -1)

        for size in self.hidden_sizes:
            x = nn.relu(nn.Dense(size)(x))

        mean = nn.Dense(self.output_dim)(x)
        log_std = nn.Dense(self.output_dim)(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (nn.tanh(log_std) + 1)
        return mean, log_std


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(self, state: Array, action: Array, time_idx: Array) -> Array:
        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, action, time_idx[..., None]], -1)

        values = [MLPDecoder(self.hidden_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
