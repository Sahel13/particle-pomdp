import jax
from jax import Array, numpy as jnp
from distrax import (
    Block,
    ScalarAffine,
    MultivariateNormalDiag
)

from ppomdp.core import (
    PRNGKey,
    TransitionModel,
    ObservationModel,
)
from ppomdp.envs.core import POMDPEnv

jax.config.update("jax_enable_x64", True)

state_dim = 1
action_dim = 1
obs_dim = 1

num_envs = 1
num_time_steps = 25
action_shift = 0.0
action_scale = 3.0
action_trans = Block(
    ScalarAffine(scale=action_scale, shift=action_shift),
    ndims=1
)


def mean_trans(s: Array, a: Array) -> Array:
    a = action_trans.forward(a)
    return s + 0.1 * a


def stddev_trans(s: Array, a: Array) -> Array:
    a = action_trans.forward(a)
    return jnp.array([0.01])


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
    return s


def stddev_obs(s: Array) -> Array:
    b = jnp.array([5.0])  # beacon position
    return 1e-2 * jnp.ones_like(s) + (b - s)**2 / 2.0


def sample_obs(rng_key: Array, s: Array) -> Array:
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
    state_cost = jax.lax.cond(
        t < num_time_steps - 1,
        lambda _: 0.,
        lambda _: jnp.dot(s, s),
        operand=None
    )
    action_cost = 1e-4 * jnp.dot(a, a)
    return - 0.5 * state_cost - 0.5 * action_cost


prior_dist = MultivariateNormalDiag(
    loc=2.0 * jnp.ones((state_dim,)),
    scale_diag=1.0 * jnp.ones((state_dim,))
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
feature_fn = lambda x: x

LightDark1DEnv = POMDPEnv(
    num_envs=num_envs,
    state_dim=state_dim,
    action_dim=action_dim,
    obs_dim=obs_dim,
    num_time_steps=num_time_steps,
    prior_dist=prior_dist,
    trans_model=trans_model,
    obs_model=obs_model,
    reward_fn=reward_fn,
    feature_fn=feature_fn
)
