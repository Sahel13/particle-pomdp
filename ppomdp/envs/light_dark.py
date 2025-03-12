"""A 2d light-dark environment."""

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import MultivariateNormalDiag
from jax import Array

from ppomdp.core import ObservationModel, TransitionModel
from ppomdp.envs.base import Environment


def mean_trans(s: Array, a: Array) -> Array:
    dt = 0.1
    return jnp.array(
        [
            s[0] + dt * s[2],
            s[1] + dt * s[3],
            s[2] + 0.5 * dt**2 * a[0],
            s[3] + 0.5 * dt**2 * a[1],
        ]
    )


def stddev_trans(s: Array, a: Array) -> Array:
    return jnp.array([1e-4, 1e-4, 1e-2, 1e-2])


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_trans(s, a), scale_diag=stddev_trans(s, a))
    return dist.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_trans(s, a), scale_diag=stddev_trans(s, a))
    return dist.log_prob(sn)


def mean_obs(s: Array) -> Array:
    H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    return H @ s


def stddev_obs(s: Array) -> Array:
    # light region along y-axis around x=5
    dist = (s[0] - 5.0) ** 2 / 2.0
    return jnp.sqrt(dist * jnp.ones(obs_dim) + 1e-4 * jnp.ones(obs_dim))


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_obs(s), scale_diag=stddev_obs(s))
    return dist.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_obs(s), scale_diag=stddev_obs(s))
    return dist.log_prob(z)


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    q = jax.lax.select(
        t < num_time_steps,
        jnp.array([1.0, 1.0, 0.0, 0.0]),
        jnp.array([50.0, 50.0, 5.0, 5.0]),
    )
    r = jnp.array([1e-5, 1e-5])
    state_cost = jnp.dot(s * q, s)
    action_cost = jnp.dot(a * r, a)
    return jax.lax.select(t == 0, 0.0, -0.5 * state_cost - 0.5 * action_cost)


num_envs = 1
state_dim = 4
action_dim = 2
obs_dim = 2
num_time_steps = 30
action_scale = 100.0
action_shift = 0.0


prior_dist = MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0, 0.0, 0.0]), scale_diag=jnp.array([2.5, 2.5, 1e-4, 1e-4])
)
# prior_dist = Deterministic(jnp.array([2.0, 2.0, 0.0, 0.0]))
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
feature_fn = lambda x: x


LightDarkTwoEnv = Environment(
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
