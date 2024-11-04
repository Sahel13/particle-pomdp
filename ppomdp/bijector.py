from typing import Tuple

import jax
from jax import Array
from jax import numpy as jnp

from distrax import Bijector


class Tanh(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return jnp.tanh(x), self.forward_log_det_jacobian(x)

    def inverse(self, y):
        x = jnp.clip(y, -0.99999997, 0.99999997)  # 0.99997 for float32
        return jnp.arctanh(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = self.inverse(y)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: Bijector) -> bool:
        return type(other) is Tanh  # pylint: disable=unidiomatic-typecheck


class Sigmoid(Bijector):
    def __init__(self):
        super().__init__(event_ndims_in=0)

    def forward_log_det_jacobian(self, x: Array) -> Array:
        return -_more_stable_softplus(-x) - _more_stable_softplus(x)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        return _more_stable_sigmoid(x), self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        x = jnp.clip(y, -0.99999997, 0.99999997)  # 0.99997 for float32
        z = jnp.log(x) - jnp.log1p(-x)
        return z, -self.forward_log_det_jacobian(z)

    def same_as(self, other: Bijector) -> bool:
        return type(other) is Sigmoid  # pylint: disable=unidiomatic-typecheck


def _more_stable_sigmoid(x: Array) -> Array:
    return jnp.where(x < -9.0, jnp.exp(x), jax.nn.sigmoid(x))


def _more_stable_softplus(x: Array) -> Array:
    return jnp.where(x < -9.0, jnp.log1p(jnp.exp(x)), jax.nn.softplus(x))