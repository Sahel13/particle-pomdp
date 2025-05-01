from typing import Callable, Union

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from ppomdp.policy.arch import (
    GRUEncoder,
    LSTMEncoder,
    MLPDecoder,
    DualHeadMLPDecoder
)
from ppomdp.core import Carry

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class PolicyNetwork(nn.Module):
    encoder: Union[LSTMEncoder, GRUEncoder]
    decoder: DualHeadMLPDecoder
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(
        self, carry: list[Carry], observation: Array
    ) -> tuple[list[Carry], Array, Array]:
        carry, y = self.encoder(carry, observation)
        mean, log_std = self.decoder(y)

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (nn.tanh(log_std) + 1)
        return carry, mean, log_std

    @property
    def dim(self):
        return self.decoder.output_dim

    def reset(self, batch_size: int) -> list[Carry]:
        return self.encoder.reset(batch_size)


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(self, state: Array, action: Array, time_idx: Array):
        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, action, time_idx[..., None]], -1)

        values = [MLPDecoder(self.hidden_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
