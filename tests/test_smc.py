import chex
import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from jax import random

from ppomdp.core import (
    InnerState,
    ObservationModel,
    OuterState,
    Policy,
    TransitionModel,
)
from ppomdp.smc import init, propagate_inner, resample_inner, reweight_inner, step


def get_inner_state(key: PRNGKey, shape: tuple[int, ...]) -> InnerState:
    key1, key2 = random.split(key)
    num_particles = shape[0]
    particles = random.normal(key1, shape)
    log_weights = random.uniform(key2, (num_particles,))
    return InnerState(
        particles=particles,
        log_weights=log_weights,
        weights=jnp.exp(log_weights - jax.nn.logsumexp(log_weights)),
        resampling_indices=jnp.arange(num_particles, dtype=jnp.int32),
    )


@pytest.mark.parametrize("seed", [0, 123])
def test_resample_inner(seed):
    rng_key = random.PRNGKey(seed)
    key1, key2 = random.split(rng_key, 2)
    num_particles = 20
    particle_dim = 3
    state = get_inner_state(key1, (num_particles, particle_dim))
    new_state = resample_inner(key2, state)

    resampling_idx = new_state.resampling_indices
    assert jnp.all(jnp.equal(new_state.particles, state.particles[resampling_idx]))
    assert jnp.all(resampling_idx >= 0) and jnp.all(resampling_idx <= num_particles - 1)
    assert jnp.all(
        jnp.equal(new_state.weights, jnp.ones_like(state.weights) / num_particles)
    )
    assert jnp.all(jnp.equal(new_state.log_weights, jnp.zeros_like(state.log_weights)))


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
    state = get_inner_state(key1, (10, 3))
    obs = random.normal(key2, (3,))
    new_state = reweight_inner(obs_model, state, obs)

    assert jnp.all(jnp.equal(new_state.particles, state.particles))
    assert jnp.all(jnp.equal(new_state.resampling_indices, state.resampling_indices))
    assert jnp.allclose(jnp.sum(new_state.weights), 1.0)


def test_nested_smc():
    """Test the nested SMC algorithm.

    This tests two things:
    1. The shapes of the leaves of the `OuterState` and `InnerState` objects.
    2. The number of times the `sample` and `log_prob` functions are traced for
    the transition and observation models.
    """
    obs_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # TODO: `init` causes an additional trace. Find out why.
    @chex.assert_max_traces(2)
    def sample_obs(rng_key, x):
        return obs_matrix @ x + random.normal(rng_key, (2,))

    @chex.assert_max_traces(1)
    def log_prob_obs(y, x):
        return -0.5 * jnp.sum(jnp.square(y - obs_matrix @ x))

    @chex.assert_max_traces(1)
    def sample_trans(rng_key, x, u):
        return x + u + random.normal(rng_key, x.shape)

    # This is not used by the filter.
    @chex.assert_max_traces(0)
    def log_prob_trans(xn, x, u):
        return -0.5 * jnp.sum(jnp.square(xn - x - u))

    chex.clear_trace_counter()

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
