"""The Pendulum-v1 environment from gymnasium:
https://gymnasium.farama.org/environments/classic_control/pendulum/."""

from functools import partial

import jax.numpy as jnp
from chex import PRNGKey
from distrax import Distribution, MultivariateNormalDiag, Normal
from jax import Array

from ppomdp.core import ObservationModel, TransitionModel
from ppomdp.slac.utils import Environment


def ode(s: Array, a: Array) -> Array:
    m, l = 1.0, 1.0
    g, d = 9.81, 1e-3

    q, dq = s[0], s[1]
    ddq = -3.0 * g / (2.0 * l) * jnp.sin(q)
    ddq += (a[0] - d * dq) * 3.0 / (m * l**2)
    return jnp.array([dq, ddq])


def step(s: Array, a: Array) -> Array:
    dt = 0.05
    return s + ode(s, a) * dt


def get_transition_model(noise_dist: Distribution) -> TransitionModel:
    def sample(rng_key: PRNGKey, s: Array, a: Array) -> Array:
        return step(s, a) + noise_dist.sample(seed=rng_key)

    def log_prob(sn: Array, s: Array, a: Array) -> Array:
        mean = step(s, a)
        return noise_dist.log_prob(sn - mean)

    return TransitionModel(sample=sample, log_prob=log_prob)


def get_observation_model(noise_dist: Distribution) -> ObservationModel:
    def get_obs(s: Array) -> Array:
        """The observation is the position of the bob in Cartesian coordinates."""
        return jnp.array((jnp.sin(s[0]), jnp.cos(s[0])))

    def sample(rng_key: PRNGKey, s: Array) -> Array:
        return get_obs(s) + noise_dist.sample(seed=rng_key)

    def log_prob(z: Array, s: Array) -> Array:
        mean_obs = get_obs(s)
        return noise_dist.log_prob(z - mean_obs)

    return ObservationModel(sample=sample, log_prob=log_prob)


def reward_fn(s: Array, a: Array, t: int) -> Array:
    def angle_normalize(x):
        return x % (2 * jnp.pi)

    cost = (
        jnp.square(angle_normalize(s[0]) - jnp.pi)
        + 0.1 * jnp.square(s[1])
        + 0.001 * jnp.square(a[0])
    )
    return -cost


num_envs = 1
state_dim = 2
action_dim = 1
obs_dim = 2
num_time_steps = 100
action_scale = 2.0
action_shift = 0.0

_default_init_state = jnp.array([0.0, 0.0])
prior_dist = Normal(_default_init_state, jnp.array([1e-16, 1e-16]))

trans_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.array([1e-4, 0.025]))
trans_model = get_transition_model(trans_noise)

obs_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.ones(2) * 1e-4)
obs_model = get_observation_model(obs_noise)


@partial(jnp.vectorize, signature="(n)->(m)")
def feature_fn(state: Array):
    return jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])


env = Environment(
    num_envs,
    state_dim,
    action_dim,
    obs_dim,
    num_time_steps,
    action_scale,
    action_shift,
    trans_model,
    obs_model,
    reward_fn,
    prior_dist,
    feature_fn,
)
