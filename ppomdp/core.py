from collections.abc import Callable
from typing import NamedTuple, Protocol

from chex import PRNGKey
from jax import Array


class SampleTransition(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array, a: Array) -> Array:
        r"""Sample from $f(s_t \mid s_{t-1}, a_{t-1})$."""


class LogProbTransition(Protocol):
    def __call__(self, sn: Array, s: Array, a: Array) -> Array:
        r"""Compute the log density of $f(s_t \mid s_{t-1}, a_{t-1})$."""


class TransitionModel(NamedTuple):
    r"""The transition kernel $f(s_t \mid s_{t-1}, a_{t-1})$."""

    sample: SampleTransition
    log_prob: LogProbTransition


class SampleObservation(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array) -> Array:
        r"""Sample from $h(z_t \mid s_t)$."""


class LogProbObservation(Protocol):
    def __call__(self, z: Array, s: Array) -> Array:
        r"""Compute the log density of $h(z_t \mid s_t)$."""


class ObservationModel(NamedTuple):
    r"""The observation model $h(z_t \mid s_t)$."""

    sample: SampleObservation
    log_prob: LogProbObservation


class Policy(NamedTuple):
    r"""The stochastic policy $\pi_\phi$.

    TODO: The functions signatures are not correct for an LSTM.
    """

    sample: Callable[[PRNGKey, Array], Array]
    log_prob: Callable[[Array, Array], Array]


class OuterState(NamedTuple):
    r"""State of the outer particle filter.

    particles: tuple[Array, Array]
        Tuple of the observations and actions $(z_t^{1:N}, a_t^{1:N})$.
    """

    particles: tuple[Array, Array]
    weights: Array
    resampling_indices: Array


class InnerState(NamedTuple):
    """State of the inner particle filter.

    particles: Array
        The state particles $s_t^{nm}$.
    """

    particles: Array
    log_weights: Array
    weights: Array
    resampling_indices: Array
