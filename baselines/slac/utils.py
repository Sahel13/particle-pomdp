from functools import partial

import jax

from distrax import Chain, MultivariateNormalDiag, Transformed
from flax.training.train_state import TrainState
from jax import Array
from jax import numpy as jnp
from jax import random

from ppomdp.core import Carry, Parameters, PRNGKey
from ppomdp.utils import custom_split
from ppomdp.envs.core import POMDPEnv

from baselines.slac.arch import PolicyNetwork


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


@partial(
    jax.jit,
    static_argnames=(
        "env_obj",
        "policy_network",
        "num_samples"
    )
)
def policy_evaluation(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    num_samples: int = 1024,
):

    def body(val, key):
        states, carry, observations, time_idx = val

        # Sample actions.
        key, action_key = random.split(key)
        carry, _, _, actions = policy_state.apply_fn(
            rng_key=action_key,
            carry=carry,
            observation=observations,
            params=policy_state.params
        )

        # Compute rewards.
        rewards = jax.vmap(env_obj.reward_fn, (0, 0, None))(states, actions, time_idx)

        # Sample next states.
        key, state_keys = custom_split(key, num_samples + 1)
        states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

        # Sample observations.
        obs_keys = random.split(key, num_samples)
        observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

        return (states, carry, observations, time_idx + 1), (states, actions, rewards)

    # Initialize.
    key, state_key = random.split(rng_key)
    init_states = env_obj.prior_dist.sample(seed=state_key, sample_shape=num_samples)

    key, obs_keys = custom_split(key, num_samples + 1)
    init_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, init_states)
    init_carry = policy_network.reset(num_samples)

    _, (states, actions, rewards) = jax.lax.scan(
        f=body,
        init=(init_states, init_carry, init_observations, 0),
        xs=random.split(key, env_obj.num_time_steps + 1)
    )
    states = jnp.concatenate([init_states[None], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))
    return expected_reward, states, actions
