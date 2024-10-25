from typing import Dict, Callable, Sequence

from jax import Array
import jax.numpy as jnp

from flax import linen as nn
from distrax import MultivariateNormalDiag
from distrax import Transformed, Block, Chain


LSTMCarry = tuple[Array, Array]


class GaussianMLP(nn.Module):
    dim: int
    layer_size: Sequence[int]
    feature_fn: Callable
    activation: Callable
    init_log_std: Callable = nn.initializers.ones
    init_kernel: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(self, s):
        log_std = self.param('log_std', self.init_log_std, self.dim)

        y = self.feature_fn(s)
        for s in self.layer_size:
            y = self.activation(nn.Dense(s, self.init_kernel)(y))
        a = nn.Dense(self.dim)(y)
        return a


class GaussianLSTM(nn.Module):
    """An LSTM policy with a dense input and output layers."""
    dim: int
    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]
    output_size: Sequence[int]
    init_log_std: Callable = nn.initializers.ones
    init_kernel: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(self, carry: LSTMCarry, s: Array) -> tuple[LSTMCarry, Array]:
        log_std = self.param('log_std', self.init_log_std, self.dim)

        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for _size in self.encoder_size:
            y = nn.relu(nn.Dense(_size, self.init_kernel)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for _size in self.recurr_size:
            carry, y = nn.LSTMCell(_size)(carry, y)

        # pass result through output layers
        for _size in self.output_size:
            y = nn.relu(nn.Dense(_size, self.init_kernel)(y))
        a = nn.Dense(self.dim)(y)
        return carry, a


def initialize_carry(model: GaussianLSTM, input_shape: tuple[int, ...]) -> LSTMCarry:
    batch_dims = input_shape[:-1]
    mem_shape = batch_dims + (model.lstm_size,) + (model.lstm_size,)  # accounts for 2 cells
    return jnp.zeros(mem_shape), jnp.zeros(mem_shape)


def policy_distribution(
    s: Array, policy: nn.Module, params: Dict, bijector: Chain
) -> Transformed:
    a = policy.apply({'params': params}, s)

    raw_dist = MultivariateNormalDiag(
        loc=a, scale_diag=jnp.exp(params['log_std'])
    )
    squashed_dist = Transformed(
        distribution=raw_dist,
        bijector=Block(bijector, ndims=1)
    )
    return squashed_dist
