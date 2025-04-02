import jax
from distrax import Chain, MultivariateNormalDiag, Transformed
from jax import Array, random
from jax import numpy as jnp

from baselines.slac.arch import PolicyNetwork
from ppomdp.core import BeliefState, Carry, Parameters, PRNGKey
from ppomdp.envs.core import POMDPEnv
from ppomdp.utils import (
    propagate_belief,
    resample_belief,
    reweight_belief,
    systematic_resampling,
)


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    carry: list[Carry],
    observation: Array,
    params: Parameters,
    network: PolicyNetwork,
    bijector: Chain,
) -> tuple[list[Carry], Array, Array, Array]:
    carry, mean, log_std = network.apply({"params": params}, carry, observation)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return carry, action, log_prob, bijector.forward(mean)


def sample_random_actions(rng_key: PRNGKey, env_obj: POMDPEnv) -> Array:
    return random.uniform(
        key=rng_key,
        shape=(env_obj.num_envs, env_obj.action_dim),
        minval=-1.0,
        maxval=1.0,
    )


def belief_init(
    rng_key: PRNGKey, env_obj: POMDPEnv, observation: Array, num_belief_particles: int
) -> BeliefState:
    """Initialize the particle filter to track the belief state."""
    particles = env_obj.prior_dist.sample(
        seed=rng_key, sample_shape=(num_belief_particles,)
    )
    log_weights = jax.vmap(env_obj.obs_model.log_prob, (None, 0))(
        observation, particles
    )
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    dummy_resampling_indices = jnp.zeros(num_belief_particles, dtype=jnp.int32)
    return BeliefState(particles, log_weights, weights, dummy_resampling_indices)


def belief_update(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    belief_state: BeliefState,
    observation: Array,
    action: Array,
) -> BeliefState:
    """Single step of the particle filter to track the belief state."""
    resample_key, propagate_key = random.split(rng_key)
    resampled_belief = resample_belief(
        resample_key, belief_state, systematic_resampling
    )
    particles = propagate_belief(
        propagate_key, env_obj.trans_model, resampled_belief.particles, action
    )
    resampled_belief = resampled_belief._replace(particles=particles)
    return reweight_belief(env_obj.obs_model, resampled_belief, observation)
