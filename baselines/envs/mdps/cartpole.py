from functools import partial

import jax
from jax import Array, numpy as jnp

from distrax import (
    Block,
    ScalarAffine,
    Deterministic,
    MultivariateNormalDiag
)

from ppomdp.core import TransitionModel
from baselines.envs.core import PRNGKey, MDPEnv


state_dim = 4
action_dim = 1

num_envs = 1
num_time_steps = 100
action_scale = 50.0
action_shift = 0.0

action_trans = Block(
    ScalarAffine(
        scale=action_scale,
        shift=action_shift
    ),
    ndims=1
)


def ode(s: Array, a: Array) -> Array:
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


def mean_trans(s: Array, a: Array) -> Array:
    dt = 0.05
    a = action_trans.forward(a)
    return s + dt * ode(s, a)


def stddev_trans(s: Array, a: Array) -> Array:
    a = action_trans.forward(a)
    return jnp.array([1e-4, 1e-4, 1e-3, 1e-3])


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_trans(s, a),
        scale_diag=stddev_trans(s, a)
    )
    return dist.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_trans(s, a),
        scale_diag=stddev_trans(s, a)
    )
    return dist.log_prob(sn)


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    def wrap_angle(s: Array) -> Array:
        x, q, xd, qd = s
        _q = q % (2 * jnp.pi)
        return jnp.hstack((x, _q, xd, qd))

    g = jnp.array([0.0, jnp.pi, 0.0, 0.0])
    h = jax.lax.select(
        t > 0,
        jnp.array([1e0, 1e1, 1e-1, 1e-1]),
        jnp.array([0., 0., 0., 0.]),
    )
    r = jnp.array([1e-3])

    s = wrap_angle(s)
    state_cost = jnp.einsum("k,kh,h->", s - g, jnp.diag(h), s - g)
    action_cost = jnp.einsum("k,kh,h->", a, jnp.diag(r), a)
    return - 0.5 * state_cost - 0.5 * action_cost


prior_dist = Deterministic(jnp.zeros(state_dim))
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)


@partial(jnp.vectorize, signature="(m)->(n)")
def feature_fn(state: Array) -> Array:
    x, q, xd, qd = state
    sin_q, cos_q = jnp.sin(q), jnp.cos(q)
    return jnp.array([x, sin_q, cos_q, xd, qd])


CartPoleMDP = MDPEnv(
    num_envs,
    state_dim,
    action_dim,
    num_time_steps,
    prior_dist,
    trans_model,
    reward_fn,
    feature_fn,
)
