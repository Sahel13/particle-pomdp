import jax
from jax import Array, numpy as jnp

from distrax import (
    Block,
    ScalarAffine,
    Deterministic,
    MultivariateNormalDiag
)

from ppomdp.core import PRNGKey, TransitionModel
from ppomdp.envs.core import MDPEnv


state_dim = 4
action_dim = 2

num_envs = 1
num_time_steps = 30
action_scale = jnp.array([100., 100.])
action_shift = jnp.array([0., 0.])

action_trans = Block(
    ScalarAffine(
        scale=action_scale,
        shift=action_shift
    ),
    ndims=1
)


def mean_trans(s: Array, a: Array) -> Array:
    dt = 0.1
    a = action_trans.forward(a)
    return jnp.array(
        [
            s[0] + dt * s[2],
            s[1] + dt * s[3],
            s[2] + 0.5 * dt**2 * a[0],
            s[3] + 0.5 * dt**2 * a[1],
        ]
    )


def stddev_trans(s: Array, a: Array) -> Array:
    a = action_trans.forward(a)
    return jnp.array([1e-4, 1e-4, 1e-2, 1e-2])


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


# def reward_fn(s: Array, a: Array, t: Array) -> Array:
#     h = jax.lax.select(
#         t > 0,
#         jnp.array([1.0, 1.0, 1e-1, 1e-1]),
#         jnp.array([0.0, 0.0, 0.0, 0.0]),
#     )
#     r = jnp.array([1e-2, 1e-2])
#     state_cost = jnp.einsum("k,kh,h->", s, jnp.diag(h), s)
#     action_cost = jnp.einsum("k,kh,h->", a, jnp.diag(r), a)
#     return -0.5 * state_cost - 0.5 * action_cost


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    h = jax.lax.select(
        t < num_time_steps,
        jnp.array([0.0, 0.0, 0.0, 0.0]),
        jnp.array([1.0, 1.0, 1e-1, 1e-1]),
    )
    r = jnp.array([1e-2, 1e-2])
    state_cost = jnp.einsum("k,kh,h->", s, jnp.diag(h), s)
    action_cost = jnp.einsum("k,kh,h->", a, jnp.diag(r), a)
    return -0.5 * state_cost - 0.5 * action_cost


prior_dist = Deterministic(jnp.array([2.0, 2.0, 0.0, 0.0]))
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
feature_fn = lambda x: x

LightDark2DMDP = MDPEnv(
    num_envs,
    state_dim,
    action_dim,
    num_time_steps,
    prior_dist,
    trans_model,
    reward_fn,
    feature_fn,
)
