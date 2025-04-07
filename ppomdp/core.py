from typing import Dict, NamedTuple, Protocol, Union, Any

import chex
from jax import Array

from flax.core import FrozenDict
from flax.training.train_state import TrainState

LSTMCarry = tuple[Array, Array]
GRUCarry = Array
Carry = Union[LSTMCarry, GRUCarry]

PRNGKey = chex.PRNGKey
Parameters = Union[Dict[str, Any], FrozenDict[str, Any]]


class HistoryParticles(NamedTuple):
    observations: Array
    actions: Array
    carry: list[Carry]
    log_probs: Array


class HistoryState(NamedTuple):
    r"""State of the history particle filter.

    Attributes:
        particles: NamedTuple of the observations, actions and carry $(z_t^{1:N}, a_t^{1:N}, c_t^{1:N})$.
        log_weights: Log weights of obervations and actions $(z_t^{1:N}, a_t^{1:N})$.
        weights: Weights of obervations and actions $(z_t^{1:N}, a_t^{1:N})$.
        resampling_indices: Resampling indicies of obervations and actions $(z_t^{1:N}, a_t^{1:N})$.
        rewards: Expected rewards of states and actions $(s_{t}^{1:N}, a_{t-1}^{1:N})$.
    """

    particles: HistoryParticles
    log_weights: Array
    weights: Array
    resampling_indices: Array
    rewards: Array


class BeliefState(NamedTuple):
    """State of the belief particle filter.

    Attributes:
        particles: The state particles $s_t^{nm}$.
        log_weights: Log weights of paticles $s_t^{nm}$.
        weights: Weights of particles $s_t^{nm}$.
        resampling_indices: Resampling indices of particles $s_t^{nm}$.
    """

    particles: Array
    log_weights: Array
    weights: Array
    resampling_indices: Array


class BeliefInfo(NamedTuple):
    ess: Array
    mean: Array
    covar: Array


class Reference(NamedTuple):
    history_particles: HistoryParticles
    belief_state: BeliefState


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


class SamplePolicy(Protocol):
    def __call__(self, rng_key: PRNGKey, s: Array, params: Parameters) -> Array:
        r"""Sample from $\pi_\phi(a_t \mid s_t)$."""


class LogProbPolicy(Protocol):
    def __call__(self, a: Array, s: Array, params: Parameters) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t \mid s_t)$."""


class Policy(NamedTuple):
    r"""The stochastic recurrent policy $\pi_\phi$."""

    sample: SamplePolicy
    log_prob: LogProbPolicy


class ResetRecurrentPolicy(Protocol):
    def __call__(self, batch_size: int) -> list[Carry]:
        r"""Reset the recurrent state of the policy."""


class SampleRecurrentPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        r"""Sample from $\pi_\phi(a_t, carry, \mid s_t)$."""


class LogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        actions: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t, carry, \mid s_t,)$."""


class PathwiseCarryRecurrentPolicy(Protocol):
    def __call__(
        self,
        init_carry: list[Carry],
        observations: Array,
        params: Parameters
    ) -> list[Carry]:
        r"""Compute the carry of $\pi_\phi(a_t \mid s_t, carry)$."""


class PathwiseLogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        particles: HistoryParticles,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t \mid s_t, carry)$."""


class SampleAndLogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        r"""Sample from $\pi_\phi(a_t, carry, \mid s_t)$ and compute its log density."""


class CarryAndLogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        action: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        r"""Compute log density of action and update carry."""


class EntropyRecurrentPolicy(Protocol):
    def __call__(
        self,
        params: Parameters,
    ) -> Array:
        r"""Compute the entropy of $\pi_\phi$."""


class InitializeRecurrentPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        input_dim: int,
        output_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        r"""Initialize the recurrent state of the policy."""


class RecurrentPolicy(NamedTuple):
    r"""The stochastic recurrent policy $\pi_\phi$."""

    dim: int
    reset: ResetRecurrentPolicy
    sample: SampleRecurrentPolicy
    log_prob: LogProbRecurrentPolicy
    pathwise_carry: PathwiseCarryRecurrentPolicy
    pathwise_log_prob: PathwiseLogProbRecurrentPolicy
    sample_and_log_prob: SampleAndLogProbRecurrentPolicy
    carry_and_log_prob: CarryAndLogProbRecurrentPolicy
    entropy: EntropyRecurrentPolicy
    init: InitializeRecurrentPolicy


class RewardFn(Protocol):
    def __call__(self, s: Array, a: Array, t: Array) -> Array:
        r"""The  reward function $r(s_t, a_t)$."""
