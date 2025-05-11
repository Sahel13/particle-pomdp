from functools import partial
from typing import NamedTuple

import jax

from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import Parameters, PRNGKey
from ppomdp.envs.core import POMDPEnv
from ppomdp.utils import custom_split
from ppomdp.smc.utils import initialize_belief, update_belief

from baselines.dsmc.arch import PolicyNetwork


class PlanState(NamedTuple):
    states: Array
    actions: Array
    time_idxs: Array
    log_weights: Array
    weights: Array
    resampling_indices: Array
    done_flags: Array


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    particles: Array,
    params: Parameters,
    network: PolicyNetwork,
    bijector: Chain,
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, particles)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return action, log_prob, bijector.forward(mean)


@partial(
    jax.jit,
    static_argnames=(
        "env_obj",
        "num_belief_particles",
        "num_samples",
    ),
)
def policy_evaluation(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    num_belief_particles: int,
    num_samples: int = 1024,
):

    def body(carry, key):
        states, beliefs, time_idx = carry

        # Sample actions using the policy
        key, action_key = random.split(key)
        _, _, actions = policy_state.apply_fn(
            rng_key=action_key, particles=beliefs.particles, params=policy_state.params
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
        next_beliefs = jax.vmap(update_belief, (0, None, None, 0, 0, 0))(
            belief_keys, env_obj.trans_model, env_obj.obs_model, beliefs, next_observations, actions
        )

        return (next_states, next_beliefs, time_idx + 1), (states, actions, rewards)

    # Initialize
    key, state_key = random.split(rng_key)
    init_states = env_obj.init_dist.sample(seed=state_key, sample_shape=num_samples)
    key, obs_keys = custom_split(key, num_samples + 1)
    init_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, init_states)

    # Initialize beliefs
    key, belief_keys = custom_split(key, num_samples + 1)
    init_beliefs = jax.vmap(initialize_belief, in_axes=(0, None, None, 0, None))(
        belief_keys, env_obj.belief_prior, env_obj.obs_model, init_observations, num_belief_particles
    )

    _, (states, actions, rewards) = jax.lax.scan(
        f=body,
        init=(init_states, init_beliefs, 0),
        xs=random.split(key, env_obj.num_time_steps + 1),
    )

    states = jnp.concatenate([init_states[None, :, :], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))

    return expected_reward, states, actions
