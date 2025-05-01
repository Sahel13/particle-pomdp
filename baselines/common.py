from typing import NamedTuple, Dict, Union

import jax
from flax.training.train_state import TrainState
from jax import Array, random, numpy as jnp

from ppomdp.core import PRNGKey, BeliefState
from ppomdp.envs import mdps, pomdps
from ppomdp.envs.core import MDPEnv, POMDPEnv
from ppomdp.smc.utils import (
    resample_belief,
    propagate_belief,
    reweight_belief,
    systematic_resampling
)


def get_mdp(env_name: str) -> MDPEnv:
    if env_name == "pendulum":
        return mdps.PendulumEnv
    elif env_name == "cartpole":
        return mdps.CartPoleEnv
    elif env_name == "light-dark-2d":
        return mdps.LightDark2DEnv
    else:
        raise NotImplementedError


def get_pomdp(env_name: str) -> POMDPEnv:
    if env_name == "pendulum":
        return pomdps.PendulumEnv
    elif env_name == "cartpole":
        return pomdps.CartPoleEnv
    elif env_name == "target-sensing":
        return pomdps.TargetEnv
    elif env_name == "light-dark-1d":
        return pomdps.LightDark1DEnv
    elif env_name == "light-dark-2d":
        return pomdps.LightDark2DEnv
    else:
        raise NotImplementedError


class JointTrainState(NamedTuple):
    policy_state: TrainState
    critic_state: TrainState
    critic_target_params: Dict


def sample_random_actions(
    rng_key: PRNGKey,
    env_obj: Union[MDPEnv, POMDPEnv],
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


def sample_hidden_states(
    rng_key: PRNGKey,
    particles: Array,
    weights: Array
) -> Array:
    batch_size, num_particles, _ = particles.shape

    def choice_fn(key, _particles, _weights):
        idx = random.choice(key, a=num_particles, p=_weights)
        return _particles[idx]

    keys = random.split(rng_key, batch_size)
    return jax.vmap(choice_fn)(keys, particles, weights)
