import distrax
import pytest

import jax
import jax.numpy as jnp
from jax import random

from chex import PRNGKey

from distrax import (
    Chain,
    ScalarAffine
)

from ppomdp.core import (
    BeliefState,
    TransitionModel,
    ObservationModel,
)
from ppomdp.smc._smc import (
    smc,
    backward_tracing
)
from ppomdp.smc.utils import resample_belief, propagate_belief, reweight_belief
from ppomdp.policy import (
    LSTM,
    create_policy,
    log_prob_policy_pathwise
)


def get_belief_state(key: PRNGKey, shape: tuple[int, ...]) -> BeliefState:
    key1, key2 = random.split(key)
    num_particles = shape[0]
    particles = random.normal(key1, shape)
    log_weights = random.normal(key2, (num_particles,))
    return BeliefState(
        particles=particles,
        log_weights=log_weights,
        weights=jnp.exp(log_weights - jax.nn.logsumexp(log_weights)),
        resampling_indices=jnp.arange(num_particles, dtype=jnp.int32),
    )


@pytest.mark.parametrize("seed", [0, 123])
@pytest.mark.parametrize("num_particles", [1, 10])
def test_resample_belief(seed, num_particles):
    rng_key = random.PRNGKey(seed)
    key1, key2 = random.split(rng_key, 2)
    num_particles = 20
    particle_dim = 3
    state = get_belief_state(key1, (num_particles, particle_dim))
    resampled_state = resample_belief(key2, state)

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
def test_propagate_belief(seed):
    def sample(rng_key, x, u):
        return x + u + random.normal(rng_key, x.shape)

    def log_prob(xn, x, u):
        return -0.5 * jnp.sum((xn - x - u) ** 2)

    trans_model = TransitionModel(sample=sample, log_prob=log_prob)
    key = random.PRNGKey(seed)

    key1, key2, key3 = random.split(key, 3)
    state_particles = random.normal(key1, (10, 3))
    action = random.normal(key2, (3,))
    new_particles = propagate_belief(key3, trans_model, state_particles, action)
    assert new_particles.shape == state_particles.shape


@pytest.mark.parametrize("seed", [0, 123])
def test_reweight_belief(seed):
    def sample(rng_key, x):
        return x + random.normal(rng_key, x.shape)

    def log_prob(y, x):
        return -0.5 * jnp.sum((y - x) ** 2)

    obs_model = ObservationModel(sample=sample, log_prob=log_prob)
    key = random.PRNGKey(seed)
    key1, key2 = random.split(key)
    state = get_belief_state(key1, (10, 3))
    obs = random.normal(key2, (3,))
    new_state = reweight_belief(obs_model, state, obs)

    assert jnp.all(jnp.equal(new_state.particles, state.particles))
    assert jnp.all(jnp.equal(new_state.resampling_indices, state.resampling_indices))
    assert jnp.allclose(jnp.sum(new_state.weights), 1.0)


def test_nested_smc():
    """Test the nested SMC algorithm.

    This tests two things:
    1. The shapes of the leaves of the `HistoryState` and `BeliefState` objects.
    2. The number of times the `sample` and `log_prob` functions are traced for
    the transition and observation models.
    """
    dim_state = 3
    dim_action = 3
    dim_obs = 2

    num_history_particles = 128
    num_belief_particles = 64
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
        encoder_size=(256, 256),
        recurr_size=(64, 64),
        output_size=(256, 256),
    )
    bijector = Chain([ScalarAffine(0.0, 1.0)])

    prior_dist = distrax.MultivariateNormalDiag(loc=jnp.zeros((dim_state,)), scale_diag=jnp.ones((dim_state,)))
    trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
    obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
    policy = create_policy(lstm, bijector)

    def reward_fn(x, u, t):
        return -0.1 * jnp.sum(x**2 + u**2)

    rng_key = random.PRNGKey(0)
    key, obs_key, param_key = random.split(rng_key, 3)
    init_carry = policy.reset(num_history_particles)
    init_obs = random.normal(obs_key, (num_history_particles, dim_obs))
    init_params = lstm.init(param_key, init_carry, init_obs)["params"]

    key, sub_key = random.split(key)
    history_states, belief_states, _, _ = smc(sub_key, num_time_steps, num_history_particles, num_belief_particles,
                                              prior_dist, policy, init_params, trans_model, obs_model, reward_fn,
                                              slew_rate_penalty=0.0, tempering=0.5)

    # Check the shapes of the leaves of `history_states`.
    assert history_states.particles[0].shape == (num_time_steps + 1, num_history_particles, dim_obs)
    assert history_states.particles[1].shape == (num_time_steps + 1, num_history_particles, dim_action)

    assert isinstance(history_states.particles[2], list)
    assert history_states.particles[2][0][0].shape == (num_time_steps + 1, num_history_particles, 64)  # [carry][LSTMCell][memory]
    assert history_states.particles[2][0][1].shape == (num_time_steps + 1, num_history_particles, 64)  # [carry][LSTMCell][output]
    assert history_states.particles[2][1][0].shape == (num_time_steps + 1, num_history_particles, 64)  # [carry][LSTMCell][memory]
    assert history_states.particles[2][1][1].shape == (num_time_steps + 1, num_history_particles, 64)  # [carry][LSTMCell][output]

    assert history_states.weights.shape == (num_time_steps + 1, num_history_particles)
    assert history_states.resampling_indices.shape == (num_time_steps + 1, num_history_particles)

    # Check the shapes of the leaves of `belief_states`.
    assert belief_states.particles.shape == (num_time_steps + 1, num_history_particles, num_belief_particles, 3)
    assert belief_states.log_weights.shape == (num_time_steps + 1, num_history_particles, num_belief_particles)
    assert belief_states.weights.shape == (num_time_steps + 1, num_history_particles, num_belief_particles)
    assert belief_states.resampling_indices.shape == (num_time_steps + 1, num_history_particles, num_belief_particles)

    # Check that the log weights and weights are finite.
    assert jnp.all(jnp.isfinite(history_states.weights))
    assert jnp.all(jnp.isfinite(belief_states.log_weights))
    assert jnp.all(jnp.isfinite(belief_states.weights))


@pytest.mark.parametrize("seed", [13, 42])
def test_policy_log_prob(seed):
    """Test the policy log probability objective."""

    # For bijectors with aggresive clipping this test will fail
    # either reduce clipping or make sure the bijector is not returning nans.

    state_dim = 3
    action_dim = 3
    obs_dim = 2

    num_history_particles = 128
    num_belief_particles = 64
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
        encoder_size=(256, 256),
        recurr_size=(64, 64),
        output_size=(256, 256),
    )
    bijector = Chain([ScalarAffine(0.0, 1.0)])
    # bijector = Chain([ScalarAffine(0.0, 1.0), Tanh()])

    prior_dist = distrax.MultivariateNormalDiag(loc=jnp.zeros((state_dim,)), scale_diag=jnp.ones((state_dim,)))
    trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
    obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
    policy = create_policy(lstm, bijector)

    def reward_fn(x, u, t):
        return -0.1 * jnp.sum(x**2 + u**2)

    rng_key = random.PRNGKey(seed)
    key, obs_key, param_key = random.split(rng_key, 3)
    init_carry = policy.reset(num_history_particles)
    init_obs = random.normal(obs_key, (num_history_particles, obs_dim))
    init_params = lstm.init(param_key, init_carry, init_obs)["params"]

    # Run SMC and plot smoothed trajectories.
    key, sub_key = random.split(rng_key)
    history_states, belief_states, belief_infos, _ = smc(sub_key, num_time_steps, num_history_particles,
                                                         num_belief_particles, prior_dist, policy, init_params,
                                                         trans_model, obs_model, reward_fn, slew_rate_penalty=0.0,
                                                         tempering=0.1)

    key, sub_key = random.split(key)
    traced_history, _, _ = backward_tracing(
        sub_key, history_states, belief_states, belief_infos
    )

    smc_log_probs = traced_history.particles.log_probs[:-1, :]
    acc_log_probs = log_prob_policy_pathwise(traced_history.particles, policy, init_params)

    assert jnp.linalg.norm(smc_log_probs - acc_log_probs) < 1e-3
