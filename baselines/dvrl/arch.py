from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.policy.arch import MLPDecoder, DualHeadMLPDecoder

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    encoding_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    output_dim: int = 1

    @nn.compact
    def __call__(self, particles: Array, weights: Array) -> tuple[Array, Array]:

        # Encode the belief state into a fixed-size vector.
        # The original DVRL implementation uses an RNN to do this, but this is
        # an arbitrary choice, since the weighted particles do not have a temporal structure.
        features = self.feature_fn(particles)
        inputs = jnp.concatenate([features, weights[..., None]], -1)
        encoding = nn.DenseGeneral(features=self.encoding_dim, axis=(-2, -1))(inputs)

        mean, log_std = DualHeadMLPDecoder(self.hidden_sizes, self.output_dim)(encoding)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (nn.tanh(log_std) + 1)
        return mean, log_std


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    encoding_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(
        self, particles: Array, weights: Array, action: Array, time_idx: Array
    ):
        # Encode the belief state into a fixed-size vector.
        features = self.feature_fn(particles)
        inputs = jnp.concatenate([features, weights[..., None]], -1)
        encoding = nn.DenseGeneral(features=self.encoding_dim, axis=(-2, -1))(inputs)

        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([encoding, action, time_idx[..., None]], -1)
        values = [MLPDecoder(self.hidden_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
