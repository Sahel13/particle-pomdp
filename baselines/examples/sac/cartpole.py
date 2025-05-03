import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from baselines.common import get_mdp
from baselines.sac import (
    SAC,
    mdp_init,
    mdp_step,
    create_train_state,
    step_and_train
)

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = SAC(
        total_time_steps=100000,
        buffer_size=100000,
        learning_starts=5000,
    )

    env_obj = get_mdp("cartpole")

    key = random.key(0)
    key, sub_key = random.split(key)
    train_state = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=config.policy_lr,
        critic_lr=config.critic_lr
    )

    key, init_key = random.split(key)
    mdp_state = mdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        policy_state=train_state.policy_state,
        random_actions=True
    )

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], mdp_state)
    buffer_obj = UniformSamplingQueue(
        max_replay_size=config.buffer_size,
        dummy_data_sample=buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, mdp_state)

    # Pre-populate the buffer with random trajectories.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        mdp_state = mdp_step(
            rng_key=sub_key,
            env_obj=env_obj,
            policy_state=train_state.policy_state,
            mdp_state=mdp_state,
            random_actions=True
        )
        buffer_state = buffer_obj.insert(buffer_state, mdp_state)

        if jnp.all(mdp_state.done_flags == 1):
            avg_return = mdp_state.total_rewards.mean()
            print(f"Step: {global_step:7d} | Average return: {avg_return:.2f}")

    # Ensure that training starts with a fresh episode.
    mdp_state = mdp_state._replace(done_flags=jnp.ones(env_obj.num_envs, dtype=jnp.int32))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env_obj.num_time_steps + 1)

    # Training loop.
    for global_step in range(
        config.learning_starts,
        config.total_time_steps,
        steps_per_epoch
    ):
        key, sub_key = random.split(key)
        mdp_state, buffer_state, train_state = \
            step_and_train(
                rng_key=sub_key,
                env_obj=env_obj,
                train_state=train_state,
                mdp_state=mdp_state,
                buffer_obj=buffer_obj,
                buffer_state=buffer_state,
                num_steps=steps_per_epoch,
                alpha=config.alpha,
                gamma=config.gamma,
                tau=config.tau
            )

        avg_return = mdp_state.total_rewards.mean()
        print(f"Step: {global_step:7d} | Average return: {avg_return:.2f}")

    # Evaluate the learned policy.
    def body_fn(carry, rng_key):
        state, time_idx = carry
        action_key, state_key = random.split(rng_key)
        _, _, action = train_state.policy_state.apply_fn(
            rng_key=action_key,
            state=state,
            time_idx=time_idx,
            params=train_state.policy_state.params,
        )
        next_state = env_obj.trans_model.sample(state_key, state, action)
        return (next_state, time_idx + 1), (next_state, action)

    key, state_key = random.split(key)
    init_state = env_obj.init_dist.sample(seed=state_key)

    _, (states, actions) = jax.lax.scan(
        f=body_fn,
        init=(init_state, 0),
        xs=random.split(key, env_obj.num_time_steps)
    )
    states = jnp.concatenate([init_state[None, ...], states], axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Simulated trajectory")

    axs[0].plot(states[:, 1])
    axs[0].set_ylabel("Angle")
    axs[0].grid(True)

    axs[1].plot(states[:, 3])
    axs[1].set_ylabel("Angular velocity")
    axs[1].grid(True)

    axs[2].plot(actions[:, 0])
    axs[2].set_ylabel("Action")
    axs[2].set_xlabel("Time")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
