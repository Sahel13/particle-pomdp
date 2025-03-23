import jax
from jax import Array, random, numpy as jnp
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import PRNGKey, Parameters
from ppomdp.envs.core import POMDPEnv
from baselines.dvrl.arch import PolicyNetwork


def sample_random_actions(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
) -> Array:
    return random.uniform(
        key=rng_key,
        shape=(env_obj.num_envs, env_obj.action_dim),
        minval=-1.0,
        maxval=1.0
    )


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    particles: Array,
    weights: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain,
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, particles, weights)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return action, log_prob, bijector.forward(mean)
