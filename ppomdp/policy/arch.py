from copy import deepcopy
from typing import Callable, Optional, Union

from jax import Array, numpy as jnp
from flax import linen as nn

from ppomdp.core import LSTMCarry, GRUCarry


class AttentionEncoder(nn.Module):
    """Attention-based encoder for processing particle sets.

    This encoder uses self-attention to process sets of particles and their weights,
    followed by a weighted pooling operation to produce a fixed-size representation.

    Attributes:
        feature_fn: Function to extract features from input particles
        hidden_size: Dimension of the hidden layers after attention
        attention_size: Dimension of the attention mechanism
        output_dim: Dimension of the final output representation
        num_heads: Number of attention heads (default: 16)
    """
    feature_fn: Callable
    hidden_size: int
    attention_size: int
    output_dim: int
    num_heads: int = 16

    @nn.compact
    def __call__(self, particles: Array, weights: Array) -> Array:
        """Process a set of particles using attention.

        Args:
            particles: Array of shape (batch_size, num_particles, particle_dim)
            weights: Array of shape (batch_size, num_particles)

        Returns:
            Array of shape (batch_size, output_dim)
        """
        # Normalize weights with numerical stability
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)

        # Embed particles
        x = self.feature_fn(particles)

        # Apply self-attention
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.attention_size,
            out_features=self.attention_size,
            broadcast_dropout=False,
            deterministic=True,
            use_bias=True,
        )(x)

        x = nn.gelu(nn.Dense(self.hidden_size)(x))
        x = nn.gelu(nn.Dense(self.hidden_size)(x))

        # Weighted pooling
        x = jnp.sum(x * weights[..., None], axis=1)

        # x = nn.relu(nn.Dense(self.hidden_size)(x))
        # x = nn.relu(nn.Dense(self.hidden_size)(x))

        # Final transformation
        x = nn.Dense(self.output_dim)(x)
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM module for processing sequences with recurrent layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        dense_sizes (tuple[int, ...]): Sizes of the dense layers before the recurrent layers.
        recurr_sizes (tuple[int, ...]): Sizes of the recurrent layers.
        use_layer_norm (bool): Whether to use layer normalization in the encoder layers.
    """

    feature_fn: Callable
    dense_sizes: tuple[int, ...]
    recurr_sizes: tuple[int, ...]
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        carry: list[LSTMCarry],
        z: Array,
        a: Optional[Array] = None
    ) -> tuple[list[LSTMCarry], Array]:

        # concat inputs and pass through features layer
        # z = jnp.concatenate([z, a], axis=-1) if a is not None else z
        y = self.feature_fn(z)

        # pass features through dense layers
        for size in self.dense_sizes:
            y = nn.Dense(size)(y)
            if self.use_layer_norm:
                y = nn.LayerNorm()(y)
            y = nn.relu(y)
        y = nn.Dense(self.recurr_sizes[0])(y)
        if self.use_layer_norm:
            y = nn.LayerNorm()(y)

        # pass encodings through recurrent layers
        next_carry = deepcopy(carry)
        for k, size in enumerate(self.recurr_sizes):
            next_carry[k], y = nn.LSTMCell(size)(carry[k], y)

        return next_carry, nn.relu(y)

    def reset(self, batch_size) -> list[LSTMCarry]:
        carry = []
        for size in self.recurr_sizes:
            mem_shape = (batch_size, size)
            c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
            carry.append((c, h))
        return carry

    @property
    def dim(self):
        return self.recurr_sizes[-1]


class GRUEncoder(nn.Module):
    """
    GRU module for processing sequences with recurrent layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        dense_sizes (tuple[int, ...]): Sizes of the dense layers before the recurrent layers.
        recurr_sizes (tuple[int, ...]): Sizes of the recurrent layers.
        use_layer_norm (bool): Whether to use layer normalization in the encoder layers.
    """

    feature_fn: Callable
    dense_sizes: tuple[int, ...]
    recurr_sizes: tuple[int, ...]
    use_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        carry: list[GRUCarry],
        z: Array,
        a: Optional[Array] = None
    ) -> tuple[list[GRUCarry], Array]:

        # concat inputs and pass through features layer
        # z = jnp.concatenate([z, a], axis=-1)
        y = self.feature_fn(z)

        # pass features through dense layers
        for size in self.dense_sizes:
            y = nn.Dense(size)(y)
            if self.use_layer_norm:
                y = nn.LayerNorm()(y)
            y = nn.relu(y)
        y = nn.Dense(self.recurr_sizes[0])(y)
        if self.use_layer_norm:
            y = nn.LayerNorm()(y)

        # pass encodings through recurrent layers
        next_carry = deepcopy(carry)
        for k, size in enumerate(self.recurr_sizes):
            next_carry[k], y = nn.GRUCell(size)(carry[k], y)

        return next_carry, nn.relu(y)

    def reset(self, batch_size) -> list[GRUCarry]:
        carry = []
        for size in self.recurr_sizes:
            mem_shape = (batch_size, size)
            h = jnp.zeros(mem_shape)  # GRUCarry
            carry.append(h)
        return carry

    @property
    def dim(self):
        return self.recurr_sizes[-1]


class MLPDecoder(nn.Module):
    """
    a standard multi layer perceptron as an action decoder

    Attributes:
        decoder_sizes (tuple[int, ...]): Sizes of the decoder layers.
        output_dim (int): Size of the output layer.
    """

    decoder_sizes: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # pass result through decoder layers
        for size in self.decoder_sizes:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.output_dim)(x)


class DualHeadMLPDecoder(nn.Module):
    decoder_sizes: tuple[int, ...]
    output_dim: int

    @nn.compact
    def __call__(self, x: Array):
        # pass result through decoder layers
        for size in self.decoder_sizes:
            x = nn.relu(nn.Dense(size)(x))
        y = nn.Dense(self.output_dim)(x)
        z = nn.Dense(self.output_dim)(x)
        return y, z


class MLPConditioner(nn.Module):
    """
    MLPConditioner is a module that conditions an input array `x` with a context array `context`
    to produce parameters for a bijector. It uses a series of hidden layers to process the input
    and context, and outputs a reshaped array suitable for use in a bijector.

    Attributes:
        event_dim (int): Dimensionality of the event space.
        hidden_sizes (tuple[int, ...]): Sizes of the hidden layers.
        num_params (int): Number of parameters per bijector.
    """

    event_dim: int
    hidden_sizes: tuple[int, ...]
    num_params: int  # number of parameters per bijector

    @nn.compact
    def __call__(self, x: Array, context: Array):
        batch_shape = x.shape[:-1]

        x = jnp.hstack([x, context])
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.event_dim * self.num_params,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )(x)

        x = x.reshape(*batch_shape, self.event_dim, self.num_params)
        return x


class LinearGaussDecoder(nn.Module):
    """
    A simple linear decoder that outputs parameters for a Gaussian distribution.

    This decoder performs a single linear transformation (matrix multiplication + bias)
    on the input and outputs both a mean vector and log standard deviation vector for
    a multivariate Gaussian distribution. The log standard deviation is stored as a
    learnable parameter.

    Attributes:
        output_dim (int): Dimensionality of the output Gaussian distribution.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
                                Defaults to ones initialization.

    The decoder architecture:
    1. Applies single linear transformation: y = Wx + b
    2. Outputs a mean vector of size output_dim
    3. Maintains a learnable log standard deviation parameter
    4. Returns both the mean and log standard deviation for the Gaussian distribution
    """

    output_dim: int
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: Array) -> tuple[Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        # Single linear transformation: y = Wx + b
        mean = nn.Dense(self.output_dim)(x)
        return mean, log_std

    @property
    def dim(self) -> int:
        return self.output_dim

    def entropy(self, log_std: Array) -> Array:
        return 0.5 * (
            self.output_dim * jnp.log(2.0 * jnp.pi * jnp.exp(1))
            + jnp.linalg.slogdet(jnp.diag(jnp.exp(2. * log_std)))[1]
        )


class NeuralGaussDecoder(nn.Module):
    """
    A neural network decoder that outputs parameters for a Gaussian distribution.

    This decoder processes input features through a series of dense layers and outputs
    both a mean vector and log standard deviation vector for a multivariate Gaussian
    distribution. The log standard deviation is stored as a learnable parameter.

    Attributes:
        decoder_sizes (tuple[int, ...]): Sizes of the hidden layers in the decoder network.
        output_dim (int): Dimensionality of the output Gaussian distribution.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
                                Defaults to ones initialization.

    The decoder architecture:
    1. Processes input through dense layers with ReLU activation
    2. Outputs a mean vector of size output_dim
    3. Maintains a learnable log standard deviation parameter
    4. Returns both the mean and log standard deviation for the Gaussian distribution
    """

    decoder_sizes: tuple[int, ...]
    output_dim: int
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: Array, z: Optional[Array] = None) -> tuple[Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_dim)

        # x = jnp.concatenate([x, z], axis=-1) if z is not None else x
        for size in self.decoder_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
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


RecurrentEncoder = Union[
    LSTMEncoder,
    GRUEncoder,
]