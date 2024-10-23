import jax
import jax.numpy as jnp
from chex import PRNGKey
from jax import Array, random

from ppomdp.core import InnerState, ObservationModel, OuterState, TransitionModel


def resample_inner(rng_key: PRNGKey, state: InnerState) -> InnerState:
    """Resample the inner particles for a single trajectory.

    Args:
        rng_key: PRNGKey
        state: InnerState
            The state associated with a single outer trajectory.
            Leaves have shape (M, ...).
    """
    num_particles = state.particles.shape[0]
    resampling_idx = random.choice(
        rng_key, num_particles, shape=(num_particles,), p=state.weights
    )
    inner_state = InnerState(
        particles=state.particles[resampling_idx],
        log_weights=jnp.zeros(num_particles),
        weights=jnp.ones(num_particles) / num_particles,
        resampling_indices=resampling_idx,
    )
    return inner_state


def propagate_inner(
    rng_key: PRNGKey,
    model: TransitionModel,
    particles: Array,
    action: Array,
) -> Array:
    r"""Propagate the inner particles for a single trajectory.

    Args:
        rng_key: PRNGKey
        model: TransitionModel
            The transition model for the state.
        particles: Array
            The state particles $\{x_{t-1}^{nm}\}_{m=1}^M$ associated with
            the n-th outer trajectory. Has shape (M, ...).
        action: Array
            The action $u_{t-1}^n$.
    """
    num_particles = particles.shape[0]
    rng_keys = random.split(rng_key, num_particles)
    return jax.vmap(model.sample, in_axes=(0, 0, None))(rng_keys, particles, action)


def reweight_inner(
    model: ObservationModel, state: InnerState, obs: Array
) -> InnerState:
    r"""Reweight the inner particles for a single outer trajectory.

    Args:
        model: ObservationModel
            The observation model.
        state: InnerState
            The inner state associated with the n-th outer trajectory.
            Leaves have shape (M, ...).
        obs: Array.
            The observation $y_t^n$.
    """
    log_weights = jax.vmap(model.log_prob, in_axes=(None, 0))(obs, state.particles)
    log_weights += state.log_weights
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    return state._replace(log_weights=log_weights, weights=weights)


def step(
    rng_key: PRNGKey,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    outer_state: OuterState,
    inner_state: InnerState,
) -> tuple[OuterState, InnerState]:
    r"""A single step of the nested SMC algorithm.

    Args:
        rng_key: PRNGKey
        trans_model: TransitionModel
            The transition model for the state, $f(x_t \mid x_{t-1}, u_{t-1})$.
        obs_model: ObservationModel
            The observation model, $g(y_t \mid x_t)$.
        outer_state: OuterState
            Leaves have shape (N, ...).
        inner_state: InnerState
            Leaves have shape (N, M, ...).
    """
    num_particles = outer_state.weights.shape[0]

    # 1. Resample the outer particles.
    key, sub_key = random.split(rng_key)
    resampling_idx = random.choice(
        sub_key, num_particles, shape=(num_particles,), p=outer_state.weights
    )
    particles = jax.tree_map(lambda x: x[resampling_idx], outer_state.particles)
    inner_state = jax.tree_map(lambda x: x[resampling_idx], inner_state)

    # 2. Resample the inner particles.
    keys = random.split(key, num_particles + 1)
    inner_state = jax.vmap(resample_inner)(keys[1:], inner_state)

    # 3. Propagate the inner particles.
    keys = random.split(keys[0], num_particles + 1)
    inner_particles = jax.vmap(propagate_inner, in_axes=(0, None, 0, 0))(
        keys[1:], trans_model, inner_state.particles, particles[1]
    )
    inner_state = inner_state._replace(particles=inner_particles)

    # 4. TODO: Propagate the outer particles.
    # 5. Reweight the inner particles.
    inner_state = jax.vmap(reweight_inner, in_axes=(None, 0, 0))(
        obs_model, inner_state, particles[0]
    )

    # 6. TODO: Reweight the outer particles.
    return outer_state, inner_state
