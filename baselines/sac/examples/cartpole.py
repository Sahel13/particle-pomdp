import jax
from jax import random, numpy as jnp

from baselines.sac.core import Config
from baselines.sac.utils import (
    init_env,
    step_env,
    create_train_state,
    step_and_train
)
from brax.training.replay_buffers import UniformSamplingQueue

import matplotlib.pyplot as plt

from baselines.envs import CartPoleEnv as env_obj


if __name__ == "__main__":
    config = Config()

    key = random.key(config.seed)
    key, sub_key = random.split(key)
    train_state = create_train_state(sub_key, env_obj, config.policy_lr, config.critic_lr)

    key, sub_key = random.split(key)
    env_state = init_env(sub_key, env_obj, train_state.policy_state, True)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], env_state)
    buffer_obj = UniformSamplingQueue(
        config.buffer_size,
        buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    key, sub_key = random.split(key)
    buffer_state = buffer_obj.init(sub_key)
    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_state = buffer_obj.insert(buffer_state, env_state)

    # Pre-populate the buffer with random trajectories.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        env_state = step_env(sub_key, env_obj, env_state, train_state.policy_state, True)
        buffer_state = buffer_obj.insert(buffer_state, env_state)
        if jnp.all(env_state.done == 1):
            print(
                f"Step: {global_step:7d} | "
                + f"Episodic reward: {env_state.total_reward.mean():10.2f}"
            )

    # Ensure that training starts with a fresh episode.
    env_state = env_state._replace(done=jnp.ones(env_obj.num_envs))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env_obj.num_time_steps + 1)

    # Training loop.
    for global_step in range(
        config.learning_starts, config.total_timesteps, steps_per_epoch
    ):
        key, sub_key = random.split(key)
        env_state, buffer_state, train_state = \
            step_and_train(
                sub_key,
                env_obj,
                env_state,
                buffer_obj,
                buffer_state,
                train_state,
                steps_per_epoch,
                config.alpha,
                config.gamma,
            )

        print(
            f"Step: {global_step + steps_per_epoch:7d} | "
            + f"Episodic reward: {env_state.total_reward.mean():10.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Evaluate the learned policy.
    key, state_key = random.split(key)
    state = env_obj.prior_dist.sample(seed=state_key)

    def body_fn(carry, rng_key):
        _action_key, _state_key = random.split(rng_key)
        _state, _time = carry
        _, _, _action = train_state.policy_state.apply_fn(
            rng_key=_action_key,
            params=train_state.policy_state.params,
            state=_state,
            time=_time
        )
        _state = env_obj.trans_model.sample(_state_key, _state, _action)
        return (_state, _time + 1), (_state, _action)

    _, (states, actions) = jax.lax.scan(
        body_fn, (state, 0), random.split(key, env_obj.num_time_steps)
    )
    states = jnp.concatenate([state[None, ...], states], axis=0)

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
