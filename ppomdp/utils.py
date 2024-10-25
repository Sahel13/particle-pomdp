from typing import Dict, Callable, Sequence

from jax import Array
import jax.numpy as jnp

from flax import linen as nn
from distrax import MultivariateNormalDiag
from distrax import Transformed, Block, Chain


class GaussianPolicy(nn.Module):
    dim: int
    layer_size: Sequence[int]
    transform: Callable
    activation: Callable
    init_log_std: Callable = nn.initializers.ones
    init_kernel: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(self, s):
        log_std = self.param('log_std', self.init_log_std, self.dim)

        y = self.transform(s)
        for s in self.layer_size:
            y = self.activation(nn.Dense(s, self.init_kernel)(y))
        a = nn.Dense(self.dim)(y)
        return a


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
