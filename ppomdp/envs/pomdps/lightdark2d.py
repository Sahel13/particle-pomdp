import jax
from jax import Array, numpy as jnp

from distrax import (
    Block,
    ScalarAffine,
    Deterministic,
    MultivariateNormalDiag,
)

from ppomdp.core import PRNGKey, TransitionModel, ObservationModel
from ppomdp.envs.core import POMDPEnv


state_dim = 4
action_dim = 2
obs_dim = 2

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
    return jnp.array([1e-2, 1e-2, 1e-1, 1e-1])


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
    H = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    return H @ s


def stddev_obs(s: Array) -> Array:
    # light region along y-axis around x=5
    dist = 5.0 * (s[0] - 5.0)**2
    return jnp.sqrt(
        dist * jnp.ones(obs_dim)
        + 1e-4 * jnp.ones(obs_dim)
    )


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
        jnp.array([0., 0., 1e-3, 1e-3]),
        jnp.array([10., 10., 1e-3, 1e-3]),
    )
    r = jnp.array([1e-3, 1e-3])
    state_cost = jnp.einsum("k,kh,h->", s, jnp.diag(h), s)
    action_cost = jnp.einsum("k,kh,h->", a, jnp.diag(r), a)
    return -0.5 * state_cost - 0.5 * action_cost


# init_dist = Deterministic(jnp.array([2.0, 2.0, 0.0, 0.0]))
init_dist = MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0, 0.0, 0.0]),
    scale_diag=jnp.array([1.0, 1.0, 1e-4, 1e-4])
)
belief_prior = MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0, 0.0, 0.0]),
    scale_diag=jnp.array([1.0, 1.0, 1e-4, 1e-4])
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
feature_fn = lambda x: x

LightDark2DEnv = POMDPEnv(
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
