import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from baselines.common import get_pomdp
from baselines.slac.config import SLAC
from baselines.slac.utils import policy_evaluation
from baselines.slac.slac import (
    create_train_state,
    gradient_step,
    pomdp_rollout,
)

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = SLAC(
        num_belief_particles=32,
        total_time_steps=1_000_000,
        buffer_size=1_000_000,
        learning_starts=50_000,
        batch_size=128
    )

    env_obj = get_pomdp("light-dark-2d")

    key = random.key(0)
    key, sub_key = random.split(key)
    train_state, policy_network, critic_networks = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=config.policy_lr,
        critic_lr=config.critic_lr
    )

    key, init_key = random.split(key)
    pomdp_states = pomdp_rollout(
        rng_key=init_key,
        env_obj=env_obj,
        policy_state=train_state.policy_state,
        policy_network=policy_network,
        num_belief_particles=config.num_belief_particles,
        random_actions=True,
    )

    buffer_entry_prototype = jax.tree.map(lambda x: x[0], pomdp_states)
    buffer_size = config.buffer_size // env_obj.num_time_steps
    buffer_obj = UniformSamplingQueue(
        max_replay_size=buffer_size,
        dummy_data_sample=buffer_entry_prototype,
        sample_batch_size=config.batch_size,
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, pomdp_states)

    for global_step in range(
        env_obj.num_time_steps,
        config.total_time_steps,
        env_obj.num_time_steps
    ):
        random_actions = global_step <= config.learning_starts

        key, sub_key = random.split(key)
        pomdp_states = pomdp_rollout(
            rng_key=sub_key,
            env_obj=env_obj,
            policy_state=train_state.policy_state,
            policy_network=policy_network,
            num_belief_particles=config.num_belief_particles,
            random_actions=random_actions
        )
        buffer_state = buffer_obj.insert(buffer_state, pomdp_states)

        if global_step > config.learning_starts:
            buffer_state, pomdp_states_batch = buffer_obj.sample(buffer_state)
            key, train_key = random.split(key)
            train_state, *_ = gradient_step(
                rng_key=train_key,
                train_state=train_state,
                policy_network=policy_network,
                pomdp_states=pomdp_states_batch,
                alpha=config.alpha,
                gamma=config.gamma,
                tau=config.tau,
            )

        if global_step % 2000 == 0:
            key, sub_key = random.split(key)
            avg_return, _, _ = policy_evaluation(
                rng_key=sub_key,
                env_obj=env_obj,
                policy_state=train_state.policy_state,
                policy_network=policy_network
            )

            print(f"Step: {global_step:6d} | Average Return: {avg_return:.2f}")

    # Evaluate the learned policy.
    def body(val, rng_key):
        state, carry, observation = val
        action_key, state_key, obs_key = random.split(rng_key, 3)
        carry, _, _, action = train_state.policy_state.apply_fn(
            rng_key=action_key,
            params=train_state.policy_state.params,
            carry=carry,
            observation=observation
        )
        state = env_obj.trans_model.sample(state_key, state, action[0])
        observation = env_obj.obs_model.sample(obs_key, state)
        return (state, carry, observation), (state, observation, action[0])

    key, state_key, obs_key = random.split(key, 3)
    init_state = env_obj.init_dist.sample(seed=state_key)
    init_observation = env_obj.obs_model.sample(obs_key, init_state)
    init_carry = policy_network.reset(1)

    keys = random.split(key, env_obj.num_time_steps)
    _, (states, _, actions) = jax.lax.scan(
        f=body,
        init=(init_state, init_carry, init_observation),
        xs=random.split(key, env_obj.num_time_steps)
    )
    states = jnp.concatenate([init_state[None, ...], states], axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Simulated trajectory")

    axs[0].plot(states[:, 0])
    axs[0].set_ylabel("Angle")
    axs[0].grid(True)

    axs[1].plot(states[:, 1])
    axs[1].set_ylabel("Angular velocity")
    axs[1].grid(True)

    axs[2].plot(actions[:, 0])
    axs[2].set_ylabel("Action")
    axs[2].set_xlabel("Time")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
