import jax
from jax import Array, random, numpy as jnp
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import PRNGKey, Parameters, BeliefState
from ppomdp.utils import (
    resample_belief,
    propagate_belief,
    reweight_belief,
    systematic_resampling
)
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


def belief_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    observation: Array,
    num_belief_particles: int
) -> BeliefState:
    """Initialize the particle filter to track the belief state."""
    particles = env_obj.prior_dist.sample(seed=rng_key, sample_shape=(num_belief_particles,))
    log_weights = jax.vmap(env_obj.obs_model.log_prob, (None, 0))(observation, particles)
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
    key, sub_key = random.split(rng_key, 2)
    resampled_belief = resample_belief(sub_key, belief_state, systematic_resampling)
    key, sub_key = random.split(key, 2)
    particles = propagate_belief(
        rng_key=sub_key,
        model=env_obj.trans_model,
        particles=resampled_belief.particles,
        action=action
    )
    resampled_belief = resampled_belief._replace(particles=particles)
    return reweight_belief(env_obj.obs_model, resampled_belief, observation)
