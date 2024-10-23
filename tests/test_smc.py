import pytest
import jax.numpy as jnp
from chex import PRNGKey
from jax import Array, random

from ppomdp.core import InnerState, ObservationModel, TransitionModel
from ppomdp.smc import propagate_inner, resample_inner, reweight_inner


@pytest.mark.parametrize("seed", [0, 123])
def test_resample_inner(seed):
    rng_key = random.PRNGKey(seed)
    key1, key2, key3 = random.split(rng_key, 3)
    num_particles = 20
    particle_dim = 3
    particles = random.normal(key1, (num_particles, particle_dim))
    weights = random.uniform(key2, (num_particles,))
    weights /= jnp.sum(weights)
    state = InnerState(
        particles=particles,
        log_weights=jnp.log(weights),
        weights=weights,
        resampling_indices=jnp.arange(num_particles, dtype=jnp.int32),
    )
    new_state = resample_inner(key3, state)

    resampling_idx = new_state.resampling_indices
    assert jnp.allclose(new_state.particles, particles[resampling_idx])
    assert jnp.all(jnp.equal(new_state.weights, jnp.ones_like(weights) / num_particles))
    assert jnp.all(jnp.equal(new_state.log_weights, jnp.zeros_like(weights)))
    assert (jnp.all(resampling_idx >= 0) and jnp.all(resampling_idx <= num_particles - 1))


@pytest.mark.parametrize("seed", [0, 123])
def test_propagate_inner(seed):
    def sample(rng_key: PRNGKey, x: Array, u: Array) -> Array:
        return x + u + random.normal(rng_key, x.shape)

    def log_prob(xn: Array, x: Array, u: Array) -> Array:
        return -0.5 * jnp.sum((xn - x - u) ** 2)

    trans_model = TransitionModel(sample=sample, log_prob=log_prob)
    key = random.PRNGKey(seed)

    key1, key2, key3 = random.split(key, 3)
    state_particles = random.normal(key1, (10, 3))
    action = random.normal(key2, (3,))
    new_particles = propagate_inner(key3, trans_model, state_particles, action)
    assert new_particles.shape == state_particles.shape


@pytest.mark.parametrize("seed", [0, 123])
def test_reweight_inner(seed):
    def sample(rng_key, x):
        return x + random.normal(rng_key, x.shape)

    def log_prob(y, x):
        return -0.5 * jnp.sum((y - x) ** 2)

    obs_model = ObservationModel(sample=sample, log_prob=log_prob)
    key = random.PRNGKey(seed)

    key1, key2 = random.split(key)
    state = InnerState(
        particles=random.normal(key1, (10, 3)),
        log_weights=jnp.zeros(10),
        weights=jnp.ones(10) / 10,
        resampling_indices=jnp.arange(10, dtype=jnp.int32),
    )
    obs = random.normal(key2, (3,))
    new_state = reweight_inner(obs_model, state, obs)

    assert jnp.all(jnp.equal(new_state.particles, state.particles))
    assert jnp.all(jnp.equal(new_state.resampling_indices, state.resampling_indices))
    assert jnp.allclose(jnp.sum(new_state.weights), 1.0)
