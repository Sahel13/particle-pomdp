from typing import Callable, Union

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.core import Carry
from ppomdp.utils import weighted_mean
from ppomdp.arch import MLPDecoder, LSTMEncoder, GRUEncoder


class PolicyNetwork(nn.Module):
    """
    Neural policy module for processing sequences with recurrent encoding and dense decoding layers.

    Attributes:
        encoder (Union[ppomdp.arch.LSTMEncoder, ppomdp.arch.GRUEncoder]): Recurrent encoder module.
        decoder (Decoder): Dense decoder module.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
    """

    encoder: Union[LSTMEncoder, GRUEncoder]
    decoder: MLPDecoder
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, carry: list[Carry], s: Array) -> tuple[list[Carry], Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.decoder.output_dim)

        carry, s = self.encoder(carry, s)
        s = self.decoder(s)
        return carry, s, log_std

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

    # @nn.compact
    # def __call__(self, particles: Array, weights: Array, action: Array, time_idx: Array):
    #     mean = weighted_mean(particles, weights)
    #     feat = self.feature_fn(mean)
    #     time_idx = time_idx / self.time_norm
    #     x = jnp.concatenate([feat, action, time_idx[..., None]], -1)
    #     values = [MLPDecoder(self.hidden_sizes, 1)(x) for _ in range(self.num_critics)]
    #     return jnp.concatenate(values, axis=-1)

    @nn.compact
    def __call__(self, state: Array, action: Array, time_idx: Array):
        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, action, time_idx[..., None]], -1)
        values = [MLPDecoder(self.hidden_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
