from functools import partial

import jax
from jax import Array, numpy as jnp

from distrax import (
    Block,
    ScalarAffine,
    Deterministic,
    MultivariateNormalDiag
)

from ppomdp.core import PRNGKey, TransitionModel, ObservationModel
from ppomdp.envs.core import POMDPEnv


state_dim = 2
action_dim = 1
obs_dim = 2

num_envs = 1
num_time_steps = 100

action_scale = 2.0
action_shift = 0.0
action_trans = Block(
    ScalarAffine(
        scale=action_scale,
        shift=action_shift
    ),
    ndims=1
)


def ode(s: Array, a: Array) -> Array:
    m, l = 1.0, 1.0
    g, d = 9.81, 1e-3

    q, dq = s[0], s[1]
    ddq = -3.0 * g / (2.0 * l) * jnp.sin(q)
    ddq += (a[0] - d * dq) * 3.0 / (m * l**2)
    return jnp.array([dq, ddq])


def mean_trans(s: Array, a: Array) -> Array:
    dt = 0.05
    a = action_trans.forward(a)
    return s + dt * ode(s, a)


def stddev_trans(s: Array, a: Array) -> Array:
    a = action_trans.forward(a)
    return jnp.array([1e-4, 1e-2])


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


def mean_obs(s: Array) -> Array:
    q, _ = s
    return jnp.array([jnp.sin(q), jnp.cos(q)])


def stddev_obs(s: Array) -> Array:
    return jnp.array([1e-2, 1e-2])


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_obs(s),
        scale_diag=stddev_obs(s)
    )
    return dist.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_obs(s),
        scale_diag=stddev_obs(s)
    )
    return dist.log_prob(z)


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    def wrap_angle(s: Array) -> Array:
        q, dq = s
        _q = q % (2 * jnp.pi)
        return jnp.hstack((_q, dq))

    g = jnp.array([jnp.pi, 0.0])
    h = jax.lax.select(
        t > 0,
        jnp.array([1e0, 1e-2]),
        jnp.array([0., 0.]),
    )
    r = jnp.array([1e-3])

    s = wrap_angle(s)
    state_cost = jnp.einsum("k,kh,h->", s - g, jnp.diag(h), s - g)
    action_cost = jnp.einsum("k,kh,h->", a, jnp.diag(r), a)
    return - 0.5 * state_cost - 0.5 * action_cost


# init_dist = Deterministic(jnp.zeros(state_dim))
init_dist = MultivariateNormalDiag(
    loc=jnp.zeros(state_dim),
    scale_diag=jnp.array([1e-2, 1e-2])
)
belief_prior = MultivariateNormalDiag(
    loc=jnp.zeros(state_dim),
    scale_diag=jnp.array([1e-2, 1e-2])
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)


@partial(jnp.vectorize, signature="(n)->(m)")
def feature_fn(state: Array) -> Array:
    q, dq = state
    sin_q, cos_q = jnp.sin(q), jnp.cos(q)
    return jnp.array([sin_q, cos_q, dq])


PendulumEnv = POMDPEnv(
    num_envs=num_envs,
    state_dim=state_dim,
    action_dim=action_dim,
    obs_dim=obs_dim,
    num_time_steps=num_time_steps,
    init_dist=init_dist,
    belief_prior=belief_prior,
    trans_model=trans_model,
    obs_model=obs_model,
    reward_fn=reward_fn,
    feature_fn=feature_fn,
)
