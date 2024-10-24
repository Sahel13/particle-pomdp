import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from jax import Array, random

from ppomdp.core import (
    InnerState,
    ObservationModel,
    OuterState,
    Policy,
    TransitionModel,
)
from ppomdp.smc import init, propagate_inner, resample_inner, reweight_inner, step


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
    assert jnp.all(resampling_idx >= 0) and jnp.all(resampling_idx <= num_particles - 1)


@pytest.mark.parametrize("seed", [0, 123])
def test_propagate_inner(seed):
    def sample(rng_key, x, u):
        return x + u + random.normal(rng_key, x.shape)

    def log_prob(xn, x, u):
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


def test_nested_smc():
    """Test the nested SMC algorithm."""
    obs_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def sample_obs(rng_key, x):
        mean = obs_matrix @ x
        return mean + random.normal(rng_key, (2,))

    def log_prob_obs(y, x):
        return -0.5 * jnp.sum(jnp.square(y - obs_matrix @ x))

    def sample_trans(rng_key, x, u):
        return x + u + random.normal(rng_key, x.shape)

    def log_prob_trans(xn, x, u):
        return -0.5 * jnp.sum(jnp.square(xn - x - u))

    trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
    obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)

    policy = Policy(
        sample=(lambda rng_key: random.normal(rng_key, (3,))),
        log_prob=(lambda x: -0.5 * jnp.sum(x**2)),
    )

    def reward_fn(x, u):
        return -0.1 * jnp.sum(x**2 + u**2)

    M = 64
    N = 128
    key = random.PRNGKey(0)
    key, sub_key = random.split(key)
    init_inner_particles = random.normal(sub_key, (N, M, 3))

    # The nested SMC algorithm.
    key, sub_key = random.split(key)
    outer_state, inner_state = init(sub_key, init_inner_particles, obs_model, policy)

    def body(carry: tuple[OuterState, InnerState], rng_key: PRNGKey):
        outer_state, inner_state = carry

        outer_state, inner_state = step(
            rng_key=rng_key,
            trans_model=trans_model,
            obs_model=obs_model,
            policy=policy,
            reward_fn=reward_fn,
            outer_state=outer_state,
            inner_state=inner_state,
            tempering=1.0,
        )

        return (outer_state, inner_state), (outer_state, inner_state)

    T = 20
    _, (outer_states, inner_states) = jax.lax.scan(
        body, (outer_state, inner_state), random.split(key, T)
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    outer_states = concat_trees(outer_state, outer_states)
    inner_states = concat_trees(inner_state, inner_states)

    # Check the shapes of the leaves of `outer_states`.
    assert outer_states.particles[0].shape == (T + 1, N, 2)
    assert outer_states.particles[1].shape == (T + 1, N, 3)
    assert outer_states.weights.shape == (T + 1, N)
    assert outer_states.resampling_indices.shape == (T + 1, N)

    # Check the shapes of the leaves of `inner_states`.
    assert inner_states.particles.shape == (T + 1, N, M, 3)
    assert inner_states.log_weights.shape == (T + 1, N, M)
    assert inner_states.weights.shape == (T + 1, N, M)
    assert inner_states.resampling_indices.shape == (T + 1, N, M)

    # Check that the log weights and weights are finite.
    assert jnp.all(jnp.isfinite(outer_states.weights))
    assert jnp.all(jnp.isfinite(inner_states.log_weights))
    assert jnp.all(jnp.isfinite(inner_states.weights))
