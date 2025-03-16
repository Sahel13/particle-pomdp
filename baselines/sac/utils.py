from jax import Array, random, numpy as jnp
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import PRNGKey, Parameters

from baselines.envs.core import MDPEnv
from baselines.sac.arch import PolicyNetwork


def sample_random_actions(
    rng_key: PRNGKey,
    env_obj: MDPEnv,
) -> Array:
    return random.uniform(
        key=rng_key,
        shape=(env_obj.num_envs, env_obj.action_dim),
        minval=-1.0,
        maxval=1.0
    )


def policy_sample(
    rng_key: PRNGKey,
    state: Array,
    time_step: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain
) -> tuple[Array, Array]:
    mean, log_std = network.apply({"params": params}, state, time_step)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    return dist.sample(seed=rng_key)


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    state: Array,
    time_step: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, state, time_step)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    actions, log_probs = dist.sample_and_log_prob(seed=rng_key)
    return actions, log_probs, bijector.forward(mean)
