import jax
from jax import Array, random, numpy as jnp
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import PRNGKey, Parameters, Carry, BeliefState
from ppomdp.utils import (
    resample_belief,
    propagate_belief,
    reweight_belief,
    systematic_resampling,
    weighted_mean
)
from ppomdp.envs.core import POMDPEnv, POMDPState, QMDPState
from baselines.slac.arch import PolicyNetwork


@jax.jit
def get_qmdp_state(pomdp_state: POMDPState) -> QMDPState:

    states = weighted_mean(
        pomdp_state.belief_states.particles,
        pomdp_state.belief_states.weights
    )
    next_states = weighted_mean(
        pomdp_state.next_belief_states.particles,
        pomdp_state.next_belief_states.weights
    )

    return QMDPState(
        states=states,
        carry=pomdp_state.carry,
        observations=pomdp_state.observations,
        actions=pomdp_state.actions,
        next_states=next_states,
        next_carry=pomdp_state.next_carry,
        next_observations=pomdp_state.next_observations,
        rewards=pomdp_state.rewards,
        time_idxs=pomdp_state.time_idxs,
        done_flags=pomdp_state.done_flags,
    )


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    carry: list[Carry],
    observation: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain,
) -> tuple[list[Carry], Array, Array, Array]:
    carry, mean, log_std = network.apply({"params": params}, carry, observation)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return carry, action, log_prob, bijector.forward(mean)


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
