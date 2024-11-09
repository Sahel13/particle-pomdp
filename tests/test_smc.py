import distrax
import pytest

import jax
import jax.numpy as jnp
from jax import Array, random

import chex
from chex import PRNGKey

from distrax import (
    Chain,
    ScalarAffine
)

from ppomdp.core import (
    InnerState,
    TransitionModel,
    ObservationModel,
)
from ppomdp.smc import (
    resample_inner,
    propagate_inner,
    reweight_inner,
    smc,
    backward_tracing
)
from ppomdp.policy import (
    LSTM,
    get_recurrent_policy,
    log_prob_policy_pathwise
)
from ppomdp.bijector import Tanh


def get_inner_state(key: PRNGKey, shape: tuple[int, ...]) -> InnerState:
    key1, key2 = random.split(key)
    num_particles = shape[0]
    particles = random.normal(key1, shape)
    log_weights = random.normal(key2, (num_particles,))
    return InnerState(
        particles=particles,
        log_weights=log_weights,
        weights=jnp.exp(log_weights - jax.nn.logsumexp(log_weights)),
        resampling_indices=jnp.arange(num_particles, dtype=jnp.int32),
    )


@pytest.mark.parametrize("seed", [0, 123])
@pytest.mark.parametrize("num_particles", [1, 10])
def test_resample_inner(seed, num_particles):
    rng_key = random.PRNGKey(seed)
    key1, key2 = random.split(rng_key, 2)
    num_particles = 20
    particle_dim = 3
    state = get_inner_state(key1, (num_particles, particle_dim))
    resampled_state = resample_inner(key2, state)

    resampling_idx = resampled_state.resampling_indices
    assert jnp.all(jnp.equal(resampled_state.particles, state.particles[resampling_idx]))
    assert jnp.all(resampling_idx >= 0) and jnp.all(resampling_idx <= num_particles - 1)
    assert jnp.all(
        jnp.equal(resampled_state.weights, jnp.ones_like(state.weights) / num_particles)
    ) or jnp.all(jnp.equal(resampled_state.weights, state.weights))
    assert jnp.all(
        jnp.equal(resampled_state.log_weights, jnp.zeros_like(state.log_weights))
    ) or jnp.all(jnp.equal(resampled_state.log_weights, state.log_weights))


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
    dim_state = 3
    dim_action = 3
    dim_obs = 2

    num_outer_particles = 128
    num_inner_particles = 64
    num_time_steps = 50

    obs_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # `init` causes an additional trace.
    # @chex.assert_max_traces(2)
    def sample_obs(rng_key, x):
        return obs_matrix @ x + random.normal(rng_key, (2,))

    # @chex.assert_max_traces(1)
    def log_prob_obs(y, x):
        return -0.5 * jnp.sum(jnp.square(y - obs_matrix @ x))

    # @chex.assert_max_traces(1)
    def sample_trans(rng_key, x, u):
        return x + u + random.normal(rng_key, x.shape)

    # This is not used by the filter.
    # @chex.assert_max_traces(0)
    def log_prob_trans(xn, x, u):
        return -0.5 * jnp.sum(jnp.square(xn - x - u))

    # chex.clear_trace_counter()

    lstm = LSTM(
        dim=dim_action,
        feature_fn=lambda x: x,
        encoder_size=[256, 256],
        recurr_size=[64, 64],
        output_size=[256, 256],
    )
    bijector = Chain([ScalarAffine(0.0, 1.0)])

    prior_dist = distrax.MultivariateNormalDiag(loc=jnp.zeros((dim_state,)), scale_diag=jnp.ones((dim_state,)))
    trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
    obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
    policy = get_recurrent_policy(lstm, bijector)

    def reward_fn(x, u):
        return -0.1 * jnp.sum(x**2 + u**2)

    rng_key = random.PRNGKey(0)
    key, obs_key, param_key = random.split(rng_key, 3)
    init_carry = policy.reset(num_outer_particles)
    init_obs = random.normal(obs_key, (num_outer_particles, dim_obs))
    init_params = lstm.init(param_key, init_carry, init_obs)["params"]

    key, sub_key = random.split(key)
    outer_states, inner_states, _ = smc(
        sub_key,
        num_outer_particles,
        num_inner_particles,
        num_time_steps,
        prior_dist,
        trans_model,
        obs_model,
        policy,
        init_params,
        reward_fn,
        tempering=0.5,
    )

    # Check the shapes of the leaves of `outer_states`.
    assert outer_states.particles[0].shape == (num_time_steps + 1, num_outer_particles, dim_obs)
    assert outer_states.particles[1].shape == (num_time_steps + 1, num_outer_particles, dim_action)

    assert isinstance(outer_states.particles[2], list)
    assert outer_states.particles[2][0][0].shape == (num_time_steps + 1, num_outer_particles, 64)  # [carry][LSTMCell][memory]
    assert outer_states.particles[2][0][1].shape == (num_time_steps + 1, num_outer_particles, 64)  # [carry][LSTMCell][output]
    assert outer_states.particles[2][1][0].shape == (num_time_steps + 1, num_outer_particles, 64)  # [carry][LSTMCell][memory]
    assert outer_states.particles[2][1][1].shape == (num_time_steps + 1, num_outer_particles, 64)  # [carry][LSTMCell][output]

    assert outer_states.weights.shape == (num_time_steps + 1, num_outer_particles)
    assert outer_states.resampling_indices.shape == (num_time_steps + 1, num_outer_particles)

    # Check the shapes of the leaves of `inner_states`.
    assert inner_states.particles.shape == (num_time_steps + 1, num_outer_particles, num_inner_particles, 3)
    assert inner_states.log_weights.shape == (num_time_steps + 1, num_outer_particles, num_inner_particles)
    assert inner_states.weights.shape == (num_time_steps + 1, num_outer_particles, num_inner_particles)
    assert inner_states.resampling_indices.shape == (num_time_steps + 1, num_outer_particles, num_inner_particles)

    # Check that the log weights and weights are finite.
    assert jnp.all(jnp.isfinite(outer_states.weights))
    assert jnp.all(jnp.isfinite(inner_states.log_weights))
    assert jnp.all(jnp.isfinite(inner_states.weights))


@pytest.mark.parametrize("seed", [13, 42])
def test_policy_log_prob(seed):
    """Test the policy log probability objective."""

    # For bijectors with aggresive clipping this test will fail
    # either reduce clipping or make sure the bijector is not returning nans.

    state_dim = 3
    action_dim = 3
    obs_dim = 2

    num_outer_particles = 128
    num_inner_particles = 64
    num_time_steps = 50

    obs_matrix = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def sample_obs(rng_key, x):
        return obs_matrix @ x + random.normal(rng_key, (2,))

    def log_prob_obs(y, x):
        return -0.5 * jnp.sum(jnp.square(y - obs_matrix @ x))

    def sample_trans(rng_key, x, u):
        return x + u + random.normal(rng_key, x.shape)

    def log_prob_trans(xn, x, u):
        return -0.5 * jnp.sum(jnp.square(xn - x - u))

    lstm = LSTM(
        dim=action_dim,
        feature_fn=lambda x: x,
        encoder_size=[256, 256],
        recurr_size=[64, 64],
        output_size=[256, 256],
    )
    bijector = Chain([ScalarAffine(0.0, 1.0)])
    # bijector = Chain([ScalarAffine(0.0, 1.0), Tanh()])

    prior_dist = distrax.MultivariateNormalDiag(loc=jnp.zeros((state_dim,)), scale_diag=jnp.ones((state_dim,)))
    trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
    obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
    policy = get_recurrent_policy(lstm, bijector)

    def reward_fn(x, u):
        return -0.1 * jnp.sum(x**2 + u**2)

    rng_key = random.PRNGKey(seed)
    key, obs_key, param_key = random.split(rng_key, 3)
    init_carry = policy.reset(num_outer_particles)
    init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
    init_params = lstm.init(param_key, init_carry, init_obs)["params"]

    # Run SMC and plot smoothed trajectories.
    key, sub_key = random.split(rng_key)
    outer_states, inner_states, _ = smc(
        sub_key,
        num_outer_particles,
        num_inner_particles,
        num_time_steps,
        prior_dist,
        trans_model,
        obs_model,
        policy,
        init_params,
        reward_fn,
        tempering=0.1,
    )

    key, sub_key = random.split(key)
    traced_outer, _ = backward_tracing(sub_key, outer_states, inner_states)

    smc_log_probs = traced_outer.particles.log_probs[:-1, :]
    acc_log_probs = log_prob_policy_pathwise(policy, init_params, traced_outer.particles)

    assert jnp.linalg.norm(smc_log_probs - acc_log_probs) < 1e-3
