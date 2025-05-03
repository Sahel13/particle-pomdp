from jax import Array, numpy as jnp
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import PRNGKey, Parameters
from baselines.sac.arch import PolicyNetwork


def policy_sample(
    rng_key: PRNGKey,
    state: Array,
    time_idx: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain
) -> tuple[Array, Array]:
    mean, log_std = network.apply({"params": params}, state, time_idx)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    return dist.sample(seed=rng_key)


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    state: Array,
    time_idx: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, state, time_idx)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    actions, log_probs = dist.sample_and_log_prob(seed=rng_key)
    return actions, log_probs, bijector.forward(mean)
