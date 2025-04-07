from functools import partial
from typing import NamedTuple

import jax

from distrax import Chain, MultivariateNormalDiag, Transformed
from flax.training.train_state import TrainState
from jax import Array
from jax import numpy as jnp
from jax import random

from ppomdp.core import PRNGKey, Parameters
from ppomdp.utils import custom_split
from ppomdp.envs.core import POMDPEnv

from baselines.dvrl.arch import PolicyNetwork
from baselines.common import belief_init, belief_update


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


@partial(jax.jit, static_argnames=("env_obj", "num_belief_particles", "num_samples"))
def policy_evaluation(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    num_belief_particles: int,
    num_samples: int = 100,
):
    
    def body(carry, key):
        states, beliefs, time_idx = carry
        
        # Sample actions using the policy
        key, action_key = random.split(key)
        _, _, actions = policy_state.apply_fn(
            rng_key=action_key,
            particles=beliefs.particles,
            weights=beliefs.weights,
            params=policy_state.params
        )

        # Compute rewards
        rewards = jax.vmap(env_obj.reward_fn, (0, 0, None))(states, actions, time_idx)
        
        # Sample next states
        key, state_keys = custom_split(key, num_samples + 1)
        next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)
        
        # Sample observations
        key, obs_keys = custom_split(key, num_samples + 1)
        next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)
        
        # Update beliefs
        key, belief_keys = custom_split(key, num_samples + 1)
        next_beliefs = jax.vmap(belief_update, (0, None, 0, 0, 0))(
            belief_keys, env_obj, beliefs, next_observations, actions
        )

        return (next_states, next_beliefs, time_idx + 1), (states, actions, rewards)
    
    # Initialize
    key, state_key = random.split(rng_key)
    init_states = env_obj.prior_dist.sample(seed=state_key, sample_shape=num_samples)
    key, obs_keys = custom_split(key, num_samples + 1)
    init_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, init_states)

    # Initialize beliefs
    key, belief_keys = custom_split(key, num_samples + 1)
    init_beliefs = jax.vmap(belief_init, in_axes=(0, None, 0, None))(
        belief_keys, env_obj, init_observations, num_belief_particles
    )

    _, (states, actions, rewards) = jax.lax.scan(
        body,
        (init_states, init_beliefs, 0),
        random.split(key, env_obj.num_time_steps + 1),
    )
    
    states = jnp.concatenate([init_states[None, :, :], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))
    
    return expected_reward, states, actions
