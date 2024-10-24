from collections.abc import Callable
from typing import NamedTuple, Protocol

from chex import PRNGKey
from jax import Array


class SampleTransition(Protocol):
    def __call__(self, rng_key: PRNGKey, x: Array, u: Array) -> Array:
        r"""Sample from $f(x_t \mid x_{t-1}, u_{t-1})$."""


class LogProbTransition(Protocol):
    def __call__(self, xn: Array, x: Array, u: Array) -> Array:
        r"""Compute the log density of $f(x_t \mid x_{t-1}, u_{t-1})$."""


class TransitionModel(NamedTuple):
    r"""The transition kernel $f(x_t \mid x_{t-1}, u_{t-1})$."""

    sample: SampleTransition
    log_prob: LogProbTransition


class SampleObservation(Protocol):
    def __call__(self, rng_key: PRNGKey, x: Array) -> Array:
        r"""Sample from $h(y_t \mid x_t)$."""


class LogProbObservation(Protocol):
    def __call__(self, y: Array, x: Array) -> Array:
        r"""Compute the log density of $h(y_t \mid x_t)$."""


class ObservationModel(NamedTuple):
    r"""The observation model $h(y_t \mid x_t)$."""

    sample: SampleObservation
    log_prob: LogProbObservation


class Policy(NamedTuple):
    r"""The stochastic policy $\pi_\phi$.

    TODO: This is just for uniform policies at the moment (or any policy
          that samples independently from the history).
    """

    sample: Callable[[PRNGKey], Array]
    log_prob: Callable[[Array], Array]


class RewardFn(Protocol):
    def __call__(self, x: Array, u: Array) -> Array:
        r"""The  reward function $r(x_t, u_t)$."""


class OuterState(NamedTuple):
    r"""State of the outer particle filter.

    particles: tuple[Array, Array]
        Tuple of the observations and actions $(y_t^{1:N}, u_t^{1:N})$.
    """

    particles: tuple[Array, Array]
    weights: Array
    resampling_indices: Array


class InnerState(NamedTuple):
    """State of the inner particle filter.

    particles: Array
        The state particles $x_t^{nm}$.
    """

    particles: Array
    log_weights: Array
    weights: Array
    resampling_indices: Array
