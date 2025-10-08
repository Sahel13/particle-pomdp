import math
from typing import Optional, Union

import jax
from jax import Array, numpy as jnp

from ppomdp.core import PRNGKey

from distrax._src.utils import math
from distrax import (
    Bijector,
    Chain,
    Block,
    Inverse,
    ScalarAffine,
    MaskedCoupling,
    Transformed
)


def _stable_sigmoid(x: Array) -> Array:
    return jnp.where(x < -9.0, jnp.exp(x), jax.nn.sigmoid(x))


def _stable_softplus(x: Array) -> Array:
    return jnp.where(x < -9.0, jnp.log1p(jnp.exp(x)), jax.nn.softplus(x))


class Tanh(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        x = jnp.clip(y, -0.99995, 0.99995)
        return jnp.arctanh(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = self.inverse(y)
        return x, -self.forward_log_det_jacobian(x)


class Sigmoid(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return -_stable_softplus(-x) - _stable_softplus(x)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return _stable_sigmoid(x), self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = jnp.clip(y, -0.99995, 0.99995)
        z = jnp.log(x) - jnp.log1p(-x)
        return z, -self.forward_log_det_jacobian(z)
