from typing import Dict, NamedTuple, Protocol, Union, Any

import chex
from jax import Array

from flax.core import FrozenDict

LSTMCarry = tuple[Array, Array]
GRUCarry = Array
Carry = Union[LSTMCarry, GRUCarry]

PRNGKey = chex.PRNGKey
Parameters = Union[Dict[str, Any], FrozenDict[str, Any]]


class HistoryParticles(NamedTuple):
    """Represents a collection of particles, each containing history information.

    This class is intended to encapsulate the history of actions, observations,
    and carry information for a system over time. It utilizes a NamedTuple to
    ensure immutability and easy access to the stored data. It is primarily used
    in systems that require tracking and tracing a timeline of events and states.

    Attributes:
        actions (Array): A collection of actions corresponding to the history of
            particles.
        carry (list[Carry]): A list containing additional carry information for
            each particle.
        observations (Array): A collection of observations associated with the
            history of particles.
    """
    actions: Array
    carry: list[Carry]
    observations: Array


class HistoryState(NamedTuple):
    """
    Represents the state of a history in a particle filter.

    This class encapsulates key components of the particle filter's history state, including
    particles, weights, resampling indices, and rewards. It is often used to track and manage
    the evolution of particles and their associated characteristics over time as the particle
    filter operates.

    Attributes:
        particles (HistoryParticles): Particles representing the state of the system being filtered.
        log_weights (Array): Logarithmic weights associated with the particles.
        weights (Array): Normalized weights of the particles, derived from the log_weights.
        resampling_indices (Array): Indices used for resampling particles during the filtering process.
        rewards (Array): Rewards assigned or accumulated for the particles during the evolution step.
    """

    particles: HistoryParticles
    log_weights: Array
    weights: Array
    resampling_indices: Array
    rewards: Array


class BeliefState(NamedTuple):
    """
    Represents the state of a belief in probabilistic inference.

    This class is utilized to structure the belief state in applications such as
    particle filters. It encapsulates arrays representing particles, their weights,
    logarithmic weights, and resampling indices. Each attribute plays a distinct
    role in defining the state and evolution of the belief system over time.

    Attributes:
        particles (Array): The array of particles representing possible states of
            the system.
        log_weights (Array): The natural logarithm of particle weights, useful for
            numerical stability in probabilistic computations.
        weights (Array): The normalized weights of particles, summing up to one.
        resampling_indices (Array): Indices representing the result of a resampling
            operation on the particles.
    """

    particles: Array
    log_weights: Array
    weights: Array
    resampling_indices: Array


class BeliefInfo(NamedTuple):
    """
    Holds information about a belief system's essential statistics, mean, and covariance.

    This class is a NamedTuple that encapsulates data related to beliefs in the context
    of probabilistic models or statistical computations. It organizes and stores
    essential statistics, mean values, and covariance matrices, which are commonly
    used in tasks like Bayesian inference, state estimation, or uncertainty quantification.

    Attributes:
        ess (Array): Essential statistics representing fundamental statistical data.
        mean (Array): Central tendencies or expected values of the belief distribution.
        covar (Array): Covariance matrix associated with the belief statistics, reflecting
            measures of variability and relationships between variables.
    """
    ess: Array
    mean: Array
    covar: Array


class Reference(NamedTuple):
    """
    Encapsulates a reference that contains historical particle data and belief state.

    This class serves as a structured container for pairing historical particle
    observations with associated belief state information, useful in probabilistic
    and statistical models.

    Attributes:
        history_particles (HistoryParticles): Historical particle data representing
            past observations or states.
        belief_state (BeliefState): The current belief state based on the historical
            particle data.
    """
    history_particles: HistoryParticles
    belief_state: BeliefState


class RewardFn(Protocol):
    def __call__(self, s: Array, a: Array, t: Array) -> Array:
        r"""The  reward function $r(s_t, a_t)$."""


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


class ResetRecurrentPolicy(Protocol):
    def __call__(self, batch_size: int) -> list[Carry]:
        r"""Reset the recurrent state of the policy."""


class SampleRecurrentPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        r"""Sample from $\pi_\phi(a_t \mid z_t, a_{t-1}, carry)$."""


class LogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        next_actions: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t, \mid z_t, a_{t-1}, carry)$."""


class PathwiseCarryRecurrentPolicy(Protocol):
    def __call__(
        self,
        actions: Array,
        observations: Array,
        params: Parameters
    ) -> list[Carry]:
        r"""Compute the carry of $\pi_\phi(a_t \mid z_t, a_{t-1}, carry)$."""


class PathwiseLogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        actions: Array,
        observations: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the pathwise log density of $\pi_\phi$."""


class SampleAndLogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array, Array]:
        r"""Sample from $\pi_\phi(a_t, \mid z_t, a_{t-1}, carry)$ and compute its log density."""


class CarryAndLogProbRecurrentPolicy(Protocol):
    def __call__(
        self,
        next_actions: Array,
        carry: list[Carry],
        actions: Array,
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
        obs_dim: int,
        action_dim: int,
        batch_dim: int,
    ) -> Parameters:
        r"""Initialize the recurrent state of the policy."""


class RecurrentPolicy(NamedTuple):
    r"""The stochastic recurrent policy $\pi_\phi$."""

    dim: int
    init: InitializeRecurrentPolicy
    reset: ResetRecurrentPolicy
    sample: SampleRecurrentPolicy
    log_prob: LogProbRecurrentPolicy
    pathwise_carry: PathwiseCarryRecurrentPolicy
    pathwise_log_prob: PathwiseLogProbRecurrentPolicy
    sample_and_log_prob: SampleAndLogProbRecurrentPolicy
    carry_and_log_prob: CarryAndLogProbRecurrentPolicy
    entropy: EntropyRecurrentPolicy


class SampleRecurrentObservation(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        r"""Sample from $q(z_{t+1} \mid a_t, carry)$."""


class LogProbRecurrentObservation(Protocol):
    def __call__(
        self,
        next_observations: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $q(z_{t+1} \mid a_t, carry)$."""


class SampleAndLogProbRecurrentObservation(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        r"""Sample from $q(z_{t+1} \mid a_t, carry)$ and compute its log density."""


class CarryAndLogProbRecurrentObservation(Protocol):
    def __call__(
        self,
        next_observations: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        r"""Compute log density of observation and update carry."""


class InitializeRecurrentObservation(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        obs_dim: int,
        action_dim: int,
        batch_dim: int,
    ) -> Parameters:
        r"""Initialize the recurrent state of the posterior over observations."""


class ResetRecurrentObservation(Protocol):
    def __call__(self, batch_size: int) -> list[Carry]:
        r"""Reset the recurrent state of the policy."""


class RecurrentObservation(NamedTuple):
    r"""The posterior distribution $q(z_{t+1} \mid a_t, carry)$."""

    dim: int
    init: InitializeRecurrentObservation
    reset: ResetRecurrentObservation
    sample: SampleRecurrentObservation
    log_prob: LogProbRecurrentObservation
    sample_and_log_prob: SampleAndLogProbRecurrentObservation
    carry_and_log_prob: CarryAndLogProbRecurrentObservation


class SampleAttentionPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array]:
        r"""Sample from $\pi_\phi(a_t \mid \{\mathbf{x}_t^i, w_t^i\}_{i=1}^N)$."""


class LogProbAttentionPolicy(Protocol):
    def __call__(
        self,
        actions: Array,
        particles: Array,
        weights: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of $\pi_\phi(a_t \mid \{\mathbf{x}_t^i, w_t^i\}_{i=1}^N)$."""


class SampleAndLogProbAttentionPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array, Array]:
        r"""Sample from $\pi_\phi(a_t \mid \{\mathbf{x}_t^i, w_t^i\}_{i=1}^N)$ and compute its log density."""


class EntropyAttentionPolicy(Protocol):
    def __call__(
        self,
        params: Parameters,
    ) -> Array:
        r"""Compute the entropy of $\pi_\phi$."""


class InitializeAttentionPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        particle_dim: int,
        action_dim: int,
        batch_size: int,
        num_particles: int,
    ) -> Parameters:
        r"""Initialize the parameters of the attention policy."""


class ResetAttentionPolicy(Protocol):
    def __call__(self, batch_size: int) -> None:
        r"""Reset the attention policy state (returns None since attention policies don't maintain state)."""


class AttentionPolicy(NamedTuple):
    r"""The stochastic attention policy $\pi_\phi$ that processes particle sets."""

    dim: int
    init: InitializeAttentionPolicy
    reset: ResetAttentionPolicy
    sample: SampleAttentionPolicy
    log_prob: LogProbAttentionPolicy
    sample_and_log_prob: SampleAndLogProbAttentionPolicy
    entropy: EntropyAttentionPolicy


class SampleLinearPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array]:
        r"""Sample from linear policy $\pi_\phi(a_t \mid \{\mathbf{x}_t^i, w_t^i\}_{i=1}^N)$ based on weighted mean."""


class LogProbLinearPolicy(Protocol):
    def __call__(
        self,
        actions: Array,
        particles: Array,
        weights: Array,
        params: Parameters
    ) -> Array:
        r"""Compute the log density of linear policy $\pi_\phi(a_t \mid \{\mathbf{x}_t^i, w_t^i\}_{i=1}^N)$."""


class SampleAndLogProbLinearPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array, Array]:
        r"""Sample from linear policy $\pi_\phi(a_t \mid \{\mathbf{x}_t^i, w_t^i\}_{i=1}^N)$ and compute its log density."""


class EntropyLinearPolicy(Protocol):
    def __call__(
        self,
        params: Parameters,
    ) -> Array:
        r"""Compute the entropy of linear policy $\pi_\phi$."""


class InitializeLinearPolicy(Protocol):
    def __call__(
        self,
        rng_key: PRNGKey,
        particle_dim: int,
        action_dim: int,
        batch_size: int,
        num_particles: int,
    ) -> Parameters:
        r"""Initialize the parameters of the linear policy."""


class ResetLinearPolicy(Protocol):
    def __call__(self, batch_size: int) -> None:
        r"""Reset the linear policy state (returns None since linear policies don't maintain state)."""


class LinearPolicy(NamedTuple):
    r"""The stochastic linear policy $\pi_\phi$ that acts on weighted particle means."""

    dim: int
    init: InitializeLinearPolicy
    reset: ResetLinearPolicy
    sample: SampleLinearPolicy
    log_prob: LogProbLinearPolicy
    sample_and_log_prob: SampleAndLogProbLinearPolicy
    entropy: EntropyLinearPolicy
