from copy import deepcopy
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
        next_carry = deepcopy(carry)
        for k, size in enumerate(self.recurr_size):
            next_carry[k], y = nn.LSTMCell(size)(carry[k], y)

        return next_carry, y

    def reset(self, batch_size) -> list[LSTMCarry]:
        carry = []
        for size in self.recurr_size:
            mem_shape = (batch_size, size)
            c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
            carry.append((c, h))
        return carry

    @property
    def dim(self):
        return self.recurr_size[-1]


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
        next_carry = deepcopy(carry)
        for k, size in enumerate(self.recurr_size):
            next_carry[k], y = nn.GRUCell(size)(carry[k], y)

        return next_carry, y

    def reset(self, batch_size) -> list[GRUCarry]:
        carry = []
        for size in self.recurr_size:
            mem_shape = (batch_size, size)
            h = jnp.zeros(mem_shape)  # GRUCarry
            carry.append(h)
        return carry

    @property
    def dim(self):
        return self.recurr_size[-1]


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


class MLPConditioner(nn.Module):
    """
    MLPConditioner is a module that conditions an input array `x` with a context array `context`
    to produce parameters for a bijector. It uses a series of hidden layers to process the input
    and context, and outputs a reshaped array suitable for use in a bijector.

    Attributes:
        event_dim (int): Dimensionality of the event space.
        hidden_size (tuple[int, ...]): Sizes of the hidden layers.
        num_params (int): Number of parameters per bijector.
    """

    event_dim: int
    hidden_size: tuple[int, ...]
    num_params: int  # number of parameters per bijector

    @nn.compact
    def __call__(self, x: Array, context: Array):
        batch_shape = x.shape[:-1]

        x = jnp.hstack([x, context])
        for size in self.hidden_size:
            x = nn.relu(nn.Dense(size)(x))
        x = nn.Dense(
            self.event_dim * self.num_params,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )(x)

        x = x.reshape(*batch_shape, self.event_dim, self.num_params)
        return x


class NeuralGaussDecoder(nn.Module):

    decoder_size: tuple[int, ...]
    output_dim: int
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        for size in self.decoder_size:
            x = nn.relu(nn.Dense(size)(x))
        y = nn.Dense(self.output_dim)(x)
        return y, log_std

    @property
    def dim(self) -> int:
        return self.output_dim

    def entropy(self, log_std: Array) -> Array:
        return 0.5 * (
            self.output_dim * jnp.log(2.0 * jnp.pi * jnp.exp(1))
            + jnp.linalg.slogdet(jnp.diag(jnp.exp(2. * log_std)))[1]
        )
