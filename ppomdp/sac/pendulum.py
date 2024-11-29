from collections.abc import Callable

import jax.numpy as jnp
from chex import PRNGKey
from distrax import MultivariateNormalDiag
from jax import Array

from ppomdp.core import TransitionModel
from ppomdp.sac.utils import Environment

state_dim = 2
action_dim = 1
num_time_steps = 100
action_scale = 2.5
action_shift = 0.0


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


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    return euler_step(pendulum_ode, s, a, 0.05) + trans_noise.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    mean = euler_step(pendulum_ode, s, a, 0.05)
    return trans_noise.log_prob(sn - mean)


def reward_fn(s: Array, a: Array, t: int) -> Array:
    Q = jnp.array([10.0, 1.0])
    R = 1e-2
    goal = jnp.array([jnp.pi, 0.0])

    def wrap_angle(q: Array) -> Array:
        return jnp.array((q[0] % (2 * jnp.pi), q[1]))

    s = wrap_angle(s)
    cost = jnp.dot((s - goal) * Q, (s - goal)) + R * jnp.dot(a, a)
    return -0.5 * cost


prior_dist = MultivariateNormalDiag(
    loc=jnp.zeros((state_dim,)), scale_diag=jnp.ones((state_dim,)) * 1e-16
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)

env = Environment(
    state_dim,
    action_dim,
    num_time_steps,
    action_scale,
    action_shift,
    trans_model,
    reward_fn,
    prior_dist,
)
