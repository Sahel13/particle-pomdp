"""The target interception problem from https://ieeexplore.ieee.org/document/1101052."""

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

state_dim = 4
action_dim = 1
obs_dim = 1

num_envs = 1
num_time_steps = 30  # 25

action_shift = 0.0
action_scale = 12.0 / 180 * jnp.pi
action_trans = Block(
    ScalarAffine(
        scale=action_scale,
        shift=action_shift
    ),
    ndims=1
)


def ode(s: Array, a: Array) -> Array:
    r"""The ODE for the target interception problem.

    Args:
        s: The state of the system is $(x, \dot{x}, y, \dot{y})$.
        a: The action is the lateral acceleration (perpendicular to the tangent).
    """
    _, xd, _, yd = s
    xdd = - yd * a[0]
    ydd = xd * a[0]
    return jnp.array([xd, xdd, yd, ydd])


def mean_trans(s: Array, a: Array) -> Array:
    dt = 1.
    a = action_trans.forward(a)
    return s + dt * ode(s, a)


def stddev_trans(s: Array, a: Array) -> Array:
    a = action_trans.forward(a)
    return jnp.array([1e-2, 1e-1, 1e-2, 1e-1])


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
    x, _, y, _ = s
    return jnp.arctan(y / x) * jnp.ones((1,))


def stddev_obs(s: Array) -> Array:
    return 10.0 * jnp.ones((1,)) * jnp.pi / 180


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
    h = jax.lax.select(
        t < num_time_steps,
        jnp.array([0., 0., 0., 0.]),
        jnp.array([0.1, 0., 0.1, 0.]),
    )
    r = jnp.array([1e-3, 1e-3])

    state_cost = jnp.einsum("k,kh,h->", s, jnp.diag(h), s)
    action_cost = jnp.einsum("k,kh,h->", a, jnp.diag(r), a)
    return - state_cost - action_cost


# init_dist = Deterministic(jnp.array([-200.0, 12.0, 100.0, -6.0]))
init_dist = MultivariateNormalDiag(
    loc=jnp.array([-200.0, 12.0, 100.0, -6.0]),
    scale_diag=jnp.array([10.0, 1e-2, 10.0, 1e-2]),
)
belief_prior = MultivariateNormalDiag(
    loc=jnp.array([-200.0, 12.0, 100.0, -6.0]),
    scale_diag=jnp.array([10.0, 1e-2, 10.0, 1e-2]),
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
feature_fn = lambda x: x

TriangulationEnv = POMDPEnv(
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