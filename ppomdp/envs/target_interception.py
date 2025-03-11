"""The target interception problem from https://ieeexplore.ieee.org/document/1101052."""

from functools import partial

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import MultivariateNormalDiag
from jax import Array

from ppomdp.core import ObservationModel, TransitionModel
from ppomdp.envs.base import Environment, euler_step

num_envs = 1
state_dim = 4
action_dim = 1
obs_dim = 1
num_time_steps = 18
action_shift = 0.0
action_scale = 9.0 / 180 * jnp.pi
prior_dist = MultivariateNormalDiag(
    loc=jnp.array([-200.0, 12.0, 100.0, -6.0]),
    scale_diag=jnp.array([1e-8, 1.0, 1e-8, 1.0]),
)


def ode(s: Array, a: Array) -> Array:
    r"""The ODE for the target interception problem.

    Args:
        s: The state of the system is $(x, \dot{x}, y, \dot{y})$.
        a: The action is the lateral acceleration (perpendicular to the tangent).
    """
    _, xd, _, yd = s
    xdd = -yd * a[0]
    ydd = xd * a[0]
    return jnp.array((xd, xdd, yd, ydd))


def get_obs(s: Array) -> Array:
    """The observation is the angle to the target."""
    return jnp.arctan(s[2] / s[0])


trans_noise = MultivariateNormalDiag(
    jnp.zeros(4), scale_diag=jnp.array([1e-4, 1e-2, 1e-4, 1e-2])
)

obs_noise = MultivariateNormalDiag(
    jnp.zeros(1),
    scale_diag=jnp.ones(1) * jnp.pi / 180,  # 1 degree standard deviation
)


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    return euler_step(ode, s, a, 1.0) + trans_noise.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    mean = euler_step(ode, s, a, 1.0)
    return trans_noise.log_prob(sn - mean)


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    return get_obs(s) + obs_noise.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    mean_obs = get_obs(s)
    return obs_noise.log_prob(z - mean_obs)


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    x, _, y, _ = s
    dist_to_target = x**2 + y**2
    penalty = 1e-1
    cost = jax.lax.select(t < num_time_steps, 0.0, penalty * dist_to_target)
    cost += 1e-2 * jnp.sum(jnp.square(a))
    return -1.0 * cost


trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)


@partial(jnp.vectorize, signature="(m)->(n)")
def feature_fn(z: jax.Array) -> jax.Array:
    return jnp.array((jnp.sin(z[0]), jnp.cos(z[0])))


TargetInterceptionEnv = Environment(
    num_envs,
    state_dim,
    action_dim,
    obs_dim,
    num_time_steps,
    action_scale,
    action_shift,
    trans_model,
    obs_model,
    reward_fn,
    prior_dist,
    feature_fn,
)
