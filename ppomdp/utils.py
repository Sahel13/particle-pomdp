from typing import Dict, Callable, Sequence

from jax import Array
import jax.numpy as jnp

from flax import linen as nn
from distrax import MultivariateNormalDiag
from distrax import Transformed, Block, Chain

from ppomdp.core import LSTMCarry


class MLP(nn.Module):
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


def mlp_distribution(
    s: Array, policy: nn.Module, params: Dict, bijector: Chain
) -> Transformed:
    a = policy.apply(params, s)

    raw_dist = MultivariateNormalDiag(
        loc=a, scale_diag=jnp.exp(params['log_std'])
    )
    squashed_dist = Transformed(
        distribution=raw_dist,
        bijector=Block(bijector, ndims=1)
    )
    return squashed_dist


class LSTM(nn.Module):
    """An LSTM policy with a dense input and output layers."""
    dim: int
    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]
    output_size: Sequence[int]
    init_log_std: Callable = nn.initializers.ones
    init_kernel: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(self, carry: list[LSTMCarry], s: Array) -> tuple[list[LSTMCarry], Array]:
        log_std = self.param('log_std', self.init_log_std, self.dim)

        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for _size in self.encoder_size:
            y = nn.relu(nn.Dense(_size, self.init_kernel)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, _size in enumerate(self.recurr_size):
            carry[k], y = nn.LSTMCell(_size)(carry[k], y)

        # pass result through output layers
        for _size in self.output_size:
            y = nn.relu(nn.Dense(_size, self.init_kernel)(y))
        a = nn.Dense(self.dim)(y)
        return carry, a


def initialize_carry(module: LSTM, batch_size: int) -> list[LSTMCarry]:
    carry = []
    for _size in module.recurr_size:
        mem_shape = (batch_size, _size)
        c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
        carry.append((c, h))
    return carry


def lstm_distribution(
    s: Array, carry: list[LSTMCarry], policy: nn.Module, params: Dict, bijector: Chain
) -> tuple[list[LSTMCarry], Transformed]:
    carry, a = policy.apply({"params": params}, carry, s)

    raw_dist = MultivariateNormalDiag(
        loc=a, scale_diag=jnp.exp(params["log_std"])
    )
    squashed_dist = Transformed(
        distribution=raw_dist,
        bijector=Block(bijector, ndims=1)
    )
    return carry, squashed_dist
