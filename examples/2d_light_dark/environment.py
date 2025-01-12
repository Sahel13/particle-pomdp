import jax
from jax import Array
import jax.numpy as jnp

from chex import PRNGKey
import distrax
from distrax import MultivariateNormalDiag

from ppomdp.core import TransitionModel, ObservationModel

jax.config.update("jax_enable_x64", True)

state_dim = 4
action_dim = 2
obs_dim = 2
num_time_steps = 30


def mean_trans(s: Array, a: Array) -> Array:
    dt = 0.1
    return jnp.array([
        s[0] + dt * s[2],
        s[1] + dt * s[3],
        s[2] + 0.5 * dt**2 * a[0],
        s[3] + 0.5 * dt**2 * a[1]
    ])


def stddev_trans(s: Array, a: Array) -> Array:
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


def mean_obs(s: Array) -> Array:
    H = jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    return H @ s


def stddev_obs(s: Array) -> Array:
    # light region along y-axis around x=5
    dist = (s[0] - 5.0)**2 / 2.0
    return jnp.sqrt(
        dist * jnp.ones(obs_dim)
        + 1e-4 * jnp.ones(obs_dim)
    )


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


def reward_fn(s: Array, a: Array, t: int) -> Array:
    q = jax.lax.cond(
        t < num_time_steps - 1,
        lambda _: jnp.zeros((state_dim,)),
        lambda _: jnp.array([250., 250., 0., 0.]),
        operand=None
    )
    r = jnp.array([1e-4, 1e-4])
    state_cost = jnp.einsum('k,kh,h->', s, jnp.diag(q), s)
    action_cost = jnp.einsum('k,kh,h->', a, jnp.diag(r), a)
    return - 0.5 * state_cost - 0.5 * action_cost


prior_dist = distrax.MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0, 0.0, 0.0]),
    scale_diag=jnp.array([2.5, 2.5, 1e-4, 1e-4])
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
