from typing import Callable

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import MultivariateNormalDiag
from jax import Array

from ppomdp.core import ObservationModel, TransitionModel

jax.config.update("jax_enable_x64", True)


def euler_step(deriv_fn: Callable, s: Array, a: Array, dt: float) -> Array:
    return s + deriv_fn(s, a) * dt


def pendulum_ode(s: Array, a: Array) -> Array:
    m, l = 1.0, 1.0
    g, d = 9.81, 1e-3

    q, dq = s[0], s[1]
    ddq = -3.0 * g / (2.0 * l) * jnp.sin(q)
    ddq += (a[0] - d * dq) * 3.0 / (m * l**2)
    return jnp.array([dq, ddq])


trans_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.array([1e-4, 0.025]))
obs_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.ones(2) * 1e-4)


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    return euler_step(pendulum_ode, s, a, 0.05) + trans_noise.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    mean = euler_step(pendulum_ode, s, a, 0.05)
    return trans_noise.log_prob(sn - mean)


def get_obs(s: Array) -> Array:
    """The observation is the position of the bob in Cartesian coordinates."""
    return jnp.array((jnp.sin(s[0]), jnp.cos(s[0])))


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    return get_obs(s) + obs_noise.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    mean_obs = get_obs(s)
    return obs_noise.log_prob(z - mean_obs)


def reward_fn(s: Array, a: Array, _: int) -> Array:
    Q = jnp.array([10.0, 1.0])
    R = 1e-2
    goal = jnp.array([jnp.pi, 0.0])

    def wrap_angle(q: Array) -> Array:
        return jnp.array((q[0] % (2 * jnp.pi), q[1]))

    s = wrap_angle(s)
    cost = jnp.dot((s - goal) * Q, (s - goal)) + R * jnp.dot(a, a)
    return -0.5 * cost


trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
