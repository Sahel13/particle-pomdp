from typing import Callable

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import MultivariateNormalDiag
from jax import Array

from ppomdp.core import ObservationModel, TransitionModel

jax.config.update("jax_enable_x64", True)


state_dim = 4
action_dim = 1
obs_dim = 2
num_time_steps = 100


def euler_step(deriv_fn: Callable, s: Array, a: Array, dt: float) -> Array:
    return s + deriv_fn(s, a) * dt


def cartpole_ode(s: Array, a: Array) -> Array:
    g = 9.81  # gravity
    l = 0.5  # pole length
    mc = 10.0  # cart mass
    mp = 1.0  # pole mass

    x, q, xd, qd = s

    sth = jnp.sin(q)
    cth = jnp.cos(q)

    xdd = (a + mp * sth * (l * qd**2 + g * cth)) / (mc + mp * sth**2)
    qdd = (-a * cth - mp * l * qd**2 * cth * sth - (mc + mp) * g * sth) / (
        l * mc + l * mp * sth**2
    )
    return jnp.hstack((xd, qd, xdd, qdd))


trans_noise = MultivariateNormalDiag(
    jnp.zeros(4), scale_diag=jnp.array([1e-4, 1e-4, 1e-3, 1e-3])
)
obs_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.ones(2) * 1e-4)


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    return euler_step(cartpole_ode, s, a, 0.05) + trans_noise.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    mean = euler_step(cartpole_ode, s, a, 0.05)
    return trans_noise.log_prob(sn - mean)


def get_obs(s: Array) -> Array:
    """The observation is the position of the cart and the angle of the bob."""
    return s[:2]


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    return get_obs(s) + obs_noise.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    mean_obs = get_obs(s)
    return obs_noise.log_prob(z - mean_obs)


def reward_fn(s: Array, a: Array, _: int) -> Array:
    x, q, xd, qd = s
    goal = jnp.array([0.0, jnp.pi, 0.0, 0.0])

    def wrap_angle(_q: float) -> float:
        return _q % (2.0 * jnp.pi)

    Q = jnp.array([2e0, 1e1, 1e-1, 1e-1])
    R = 5e-3

    _state = jnp.array((x, wrap_angle(q), xd, qd))
    _state -= goal
    cost = jnp.dot(_state * Q, _state) + R * jnp.dot(a, a)
    return -0.5 * cost


prior_dist = MultivariateNormalDiag(
    loc=jnp.zeros((state_dim,)),
    scale_diag=jnp.ones((state_dim,)) * 1e-16
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
