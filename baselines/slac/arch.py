from typing import Callable, Union

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.core import Carry, LSTMCarry, GRUCarry


class LSTMEncoder(nn.Module):
    """
    LSTM module for processing sequences with recurrent layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (tuple[int, ...]): Sizes of the encoding layers.
        recurr_size (tuple[int, ...]): Sizes of the recurrent layers.
    """

    feature_fn: Callable
    encoder_size: tuple[int, ...]
    recurr_size: tuple[int, ...]

    @nn.compact
    def __call__(self, carry: list[LSTMCarry], s: Array) -> tuple[list[LSTMCarry], Array]:
        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for size in self.encoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, size in enumerate(self.recurr_size):
            carry[k], y = nn.LSTMCell(size)(carry[k], y)

        return carry, y

    def reset(self, batch_size) -> list[LSTMCarry]:
        carry = []
        for size in self.recurr_size:
            mem_shape = (batch_size, size)
            c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
            carry.append((c, h))
        return carry


class GRUEncoder(nn.Module):
    """
    GRU module for processing sequences with recurrent layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (tuple[int, ...]): Sizes of the encoding layers.
        recurr_size (tuple[int, ...]): Sizes of the recurrent layers.
    """

    feature_fn: Callable
    encoder_size: tuple[int, ...]
    recurr_size: tuple[int, ...]

    @nn.compact
    def __call__(self, carry: list[GRUCarry], s: Array) -> tuple[list[GRUCarry], Array]:
        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for size in self.encoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, size in enumerate(self.recurr_size):
            carry[k], y = nn.GRUCell(size)(carry[k], y)

        return carry, y

    def reset(self, batch_size) -> list[GRUCarry]:
        carry = []
        for size in self.recurr_size:
            mem_shape = (batch_size, size)
            h = jnp.zeros(mem_shape)  # GRUCarry
            carry.append(h)
        return carry


class MLPDecoder(nn.Module):
    """
    a standard multi layer perceptron as an action decoder

    Attributes:
        decoder_size (tuple[int, ...]): Sizes of the decoder layers.
        output_dim (int): Size of the output layer.
    """

    decoder_size: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # pass result through decoder layers
        for size in self.decoder_size:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.output_dim)(x)


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
    layer_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(self, state: Array, action: Array, time_idx: Array):
        feat = self.feature_fn(state)
        time_idx = time_idx / self.time_norm
        x = jnp.concatenate([feat, action, time_idx[..., None]], -1)
        values = [MLPDecoder(self.layer_sizes, 1)(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)
