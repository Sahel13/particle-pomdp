from typing import Callable, Union

from flax import linen as nn
from jax import Array
from jax import numpy as jnp

from ppomdp.policy.arch import GRUEncoder, LSTMEncoder, MLPDecoder
from ppomdp.core import Carry


class PolicyNetwork(nn.Module):
    encoder: Union[LSTMEncoder, GRUEncoder]
    decoder: MLPDecoder
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(
        self, carry: list[Carry], observation: Array
    ) -> tuple[list[Carry], Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.decoder.output_dim)

        carry, y = self.encoder(carry, observation)
        y = self.decoder(y)
        return carry, y, log_std

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
