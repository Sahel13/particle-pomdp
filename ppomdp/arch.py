from typing import Callable, Sequence

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.core import (
    LSTMCarry,
    GRUCarry,
    Carry
)


class LSTMEncoder(nn.Module):
    """
    LSTM module for processing sequences with recurrent layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
    """

    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]

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
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
    """

    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]

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
        decoder_size (Sequence[int]): Sizes of the decoder layers.
        output_dim (int): Size of the output layer.
    """

    decoder_size: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> tuple[list[Carry], Array]:
        # pass result through decoder layers
        for size in self.decoder_size:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.output_dim)(x)


class MLPConditioner(nn.Module):
    """
    MLPConditioner is a module that conditions an input array `x` with a context array `context`
    to produce parameters for a bijector. It uses a series of hidden layers to process the input
    and context, and outputs a reshaped array suitable for use in a bijector.

    Attributes:
        event_dim (int): Dimensionality of the event space.
        hidden_size (Sequence[int]): Sizes of the hidden layers.
        num_params (int): Number of parameters per bijector.
    """

    event_dim: int
    hidden_size: Sequence[int]
    num_params: int  # number of parameters per bijector

    @nn.compact
    def __call__(self, x: Array, context: Array):
        batch_shape = x.shape[:-1]

        x = jnp.hstack([x, context])
        for size in self.hidden_size:
            x = nn.relu(nn.Dense(size)(x))
        x = nn.Dense(
            self.event_dim * self.num_params,
            # kernel_init=nn.initializers.zeros,
            # bias_init=nn.initializers.zeros
        )(x)

        x = x.reshape(*batch_shape, self.event_dim, self.num_params)
        return x
