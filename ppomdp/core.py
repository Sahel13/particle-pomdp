from typing import Dict, NamedTuple, Protocol

from chex import PRNGKey
from jax import Array

from ppomdp.utils import LSTMCarry


class SampleTransition(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array, a: Array) -> Array:
        r"""Sample from $f(s_t \mid s_{t-1}, a_{t-1})$."""


class LogProbTransition(Protocol):
    def __call__(self, sn: Array, s: Array, a: Array) -> float:
        r"""Compute the log density of $f(s_t \mid s_{t-1}, a_{t-1})$."""


class TransitionModel(NamedTuple):
    r"""The transition kernel $f(s_t \mid s_{t-1}, a_{t-1})$."""

    sample: SampleTransition
    log_prob: LogProbTransition


class SampleObservation(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array) -> Array:
        r"""Sample from $h(z_t \mid s_t)$."""


class LogProbObservation(Protocol):
    def __call__(self, z: Array, s: Array) -> float:
        r"""Compute the log density of $h(z_t \mid s_t)$."""


class ObservationModel(NamedTuple):
    r"""The observation model $h(z_t \mid s_t)$."""

    sample: SampleObservation
    log_prob: LogProbObservation


class SamplePolicy(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array, params: Dict) -> Array:
        r"""Sample from $\pi_\phi(a_t \mid s_t)$."""


class LogProbPolicy(Protocol):
    def __call__(self, a: Array, s: Array, params: Dict) -> float:
        r"""Compute the log density of $\pi_\phi(a_t \mid s_t)$."""


class Policy(NamedTuple):
    r"""The stochastic recurrent policy $\pi_\phi$."""

    sample: SamplePolicy
    log_prob: LogProbPolicy


class SampleRecurrentPolicy(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array, params: Dict, carry: LSTMCarry) -> Array:
        r"""Sample from $\pi_\phi(a_t \mid s_t, carry)$."""


class LogProbRecurrentPolicy(Protocol):
    def __call__(self, a: Array, s: Array, params: Dict, carry: LSTMCarry) -> float:
        r"""Compute the log density of $\pi_\phi(a_t \mid s_t, carry)$."""


class RecurrentPolicy(NamedTuple):
    r"""The stochastic recurrent policy $\pi_\phi$."""

    sample: SampleRecurrentPolicy
    log_prob: LogProbRecurrentPolicy


class RewardFn(Protocol):
    def __call__(self, x: Array, u: Array) -> Array:
        r"""The  reward function $r(x_t, u_t)$."""


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
