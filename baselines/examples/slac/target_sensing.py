import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from baselines.common import get_pomdp
from baselines.slac import (
    SLAC,
    create_train_state,
    gradient_step,
    pomdp_rollout,
    policy_evaluation,
)

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = SLAC(
        num_belief_particles=32,
        total_time_steps=1_000_000,
        buffer_size=1_000_000,
        learning_starts=50_000,
    )

    env_obj = get_pomdp("target-sensing")

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
