from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.policy.arch import MLPDecoder


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    recurr_size: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, particles: Array, weights: Array) -> tuple[Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        # Use the weighted particle set to predict actions.
        # The original DVRL implementation uses an RNN to encode the belief state into a fixed size vector,
        # but this is an arbitrary choice, since the weighted particles do not have a temporal structure.
        features = self.feature_fn(particles)
        x = jnp.concatenate([features, weights[..., None]], -1)

        x = nn.DenseGeneral(features=self.hidden_sizes[0], axis=(-2, -1))(x)
        x = nn.relu(x)
        return MLPDecoder(self.hidden_sizes[1:], self.output_dim)(x), log_std


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    recurr_size: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(self, particles: Array, weights: Array, action: Array, time_idx: Array):
        # Encode the belief state into a fixed-size vector.
        features = self.feature_fn(particles)
        inputs = jnp.concatenate([features, weights[..., None]], -1)
        encoding = nn.RNN(nn.GRUCell(self.recurr_size))(inputs)
        encoding = jnp.take(encoding, -1, axis=-2)

        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([encoding, action, time_idx[..., None]], -1)
        values = [MLPDecoder(self.hidden_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
