import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
        total_time_steps=25000,
        buffer_size=25000,
        learning_starts=5000,
    )

    env_obj = get_mdp("pendulum")

    total_time_steps = config.total_time_steps
    buffer_size = config.buffer_size
    learning_starts = config.learning_starts
    policy_lr = config.policy_lr
    critic_lr = config.critic_lr
    batch_size = config.batch_size
    alpha = config.alpha
    gamma = config.gamma
    tau = config.tau

    key = random.key(0)
    key, sub_key = random.split(key)
    train_state = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=policy_lr,
        critic_lr=critic_lr
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
        max_replay_size=buffer_size,
        dummy_data_sample=buffer_entry_prototype,
        sample_batch_size=batch_size
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
            print(
                f"Step: {global_step:7d} | "
                + f"Episodic reward: {mdp_state.total_rewards.mean():10.2f}"
            )

    # Ensure that training starts with a fresh episode.
    mdp_state = mdp_state._replace(done_flags=jnp.ones(env_obj.num_envs, dtype=jnp.int32))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env_obj.num_time_steps + 1)

    # Training loop.
    for global_step in range(
        config.learning_starts, config.total_time_steps, steps_per_epoch
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
                alpha=alpha,
                gamma=gamma,
                tau=tau
            )

        print(
            f"Step: {global_step + steps_per_epoch:7d} | "
            + f"Episodic reward: {mdp_state.total_rewards.mean():10.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Evaluate the learned policy.
    key, state_key = random.split(key)
    state = env_obj.prior_dist.sample(seed=state_key)

    def body_fn(carry, rng_key):
        _state, _time_idx = carry
        _action_key, _state_key = random.split(rng_key)
        _, _, _action = train_state.policy_state.apply_fn(
            rng_key=_action_key,
            state=_state,
            time_idx=_time_idx,
            params = train_state.policy_state.params,
        )
        _state = env_obj.trans_model.sample(_state_key, _state, _action)
        return (_state, _time_idx + 1), (_state, _action)

    _, (states, actions) = jax.lax.scan(
        body_fn, (state, 0), random.split(key, env_obj.num_time_steps)
    )
    states = jnp.concatenate([state[None, ...], states], axis=0)

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
