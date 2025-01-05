"""The Pendulum-v1 environment from gymnasium:
https://gymnasium.farama.org/environments/classic_control/pendulum/."""

from functools import partial

import jax.numpy as jnp
from chex import PRNGKey
from distrax import MultivariateNormalDiag, Uniform
from jax import Array

from ppomdp.core import TransitionModel
from ppomdp.sac.utils import Environment

state_dim = 2
action_dim = 1
num_time_steps = 200
action_scale = 2.0
action_shift = 0.0


def step(s: Array, a: Array) -> Array:
    m, l = 1.0, 1.0
    g = 10.0
    max_speed = 8.0
    dt = 0.05
    max_torque = 2.0

    th, thdot = s[0], s[1]
    a = jnp.clip(a[0], -max_torque, max_torque)
    newthdot = thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l**2) * a) * dt
    newthdot = jnp.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt
    return jnp.array([newth, newthdot])


@partial(jnp.vectorize, signature="(n)->(m)")
def obs_fn(state: Array):
    return jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])


trans_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.array([1e-8, 1e-8]))


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    return step(s, a) + trans_noise.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    mean = step(s, a)
    return trans_noise.log_prob(sn - mean)


def reward_fn(s: Array, a: Array, t: int) -> Array:
    def angle_normalize(x):
        return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    cost = (
        jnp.square(angle_normalize(s[0]))
        + 0.1 * jnp.square(s[1])
        + 0.001 * jnp.square(a[0])
    )
    return -cost


_default_init_state = jnp.array([jnp.pi, 1.0])
prior_dist = Uniform(low=-_default_init_state, high=_default_init_state)
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
    obs_fn,
)
