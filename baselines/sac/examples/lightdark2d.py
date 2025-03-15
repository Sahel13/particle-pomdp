from functools import partial

import jax
from jax import random, numpy as jnp

from baselines.sac.core import Config
from baselines.sac.utils import init_env, step_env, create_train_state, gradient_step
from brax.training.replay_buffers import UniformSamplingQueue

import matplotlib.pyplot as plt

from baselines.envs import LightDark2DEnv as env


if __name__ == "__main__":
    config = Config()

    key = random.key(config.seed)
    key, sub_key = random.split(key)
    train_state = create_train_state(sub_key, env, config.policy_lr, config.critic_lr)

    key, sub_key = random.split(key)
    env_state = init_env(sub_key, env, train_state.policy_state, True)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], env_state)
    buffer = UniformSamplingQueue(
        config.buffer_size,
        buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )


    @partial(jax.jit, static_argnames="num_steps")
    def step_and_train(
            rng_key,
            init_train_state,
            init_env_state,
            init_buffer_state,
            num_steps
    ):

        def body(carry, rng_keys):
            _train_state, _env_state, _buffer_state = carry
            _step_key, _train_key = rng_keys

            _env_state = step_env(_step_key, env, _env_state, _train_state.policy_state)
            _buffer_state = buffer.insert(_buffer_state, env_state)

            _buffer_state, _env_state_sample = buffer.sample(buffer_state)
            _train_state, _, _ = \
                gradient_step(
                    rng_key=_train_key,
                    train_state=_train_state,
                    env_state=_env_state_sample,
                    alpha=config.alpha,
                    gamma=config.gamma
                )

            return (_train_state, _env_state, _buffer_state), None

        return jax.lax.scan(
            body,
            (init_train_state, init_env_state, init_buffer_state),
            random.split(rng_key, num_steps * 2).reshape((num_steps, 2)),
        )[0]


    key, sub_key = random.split(key)
    buffer_state = buffer.init(sub_key)
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer_state = buffer.insert(buffer_state, env_state)

    # Pre-populate the buffer with random trajectories.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        env_state = step_env(sub_key, env, env_state, train_state.policy_state, True)
        buffer_state = buffer.insert(buffer_state, env_state)
        if jnp.all(env_state.done == 1):
            print(
                f"Step: {global_step:7d} | "
                + f"Episodic reward: {env_state.total_reward.mean():10.2f}"
            )

    # Ensure that training starts with a fresh episode.
    env_state = env_state._replace(done=jnp.ones(env.num_envs))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env.num_time_steps + 1)

    # Training loop.
    for global_step in range(
            config.learning_starts, config.total_timesteps, steps_per_epoch
    ):
        key, sub_key = random.split(key)
        train_state, env_state, buffer_state = \
            step_and_train(sub_key, train_state, env_state, buffer_state, steps_per_epoch)

        print(
            f"Step: {global_step + steps_per_epoch:7d} | "
            + f"Episodic reward: {env_state.total_reward.mean():10.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Evaluate the learned policy.
    key, state_key = random.split(key)
    state = env.prior_dist.sample(seed=state_key)


    def body_fn(carry, rng_key):
        _action_key, _state_key = random.split(rng_key)
        _state, _time = carry
        _, _, _action = train_state.policy_state.apply_fn(
            rng_key=_action_key,
            params=train_state.policy_state.params,
            state=_state,
            time=_time
        )
        _state = env.trans_model.sample(_state_key, _state, _action)
        return (_state, _time + 1), (_state, _action)


    _, (states, actions) = jax.lax.scan(
        body_fn, (state, 0), random.split(key, env.num_time_steps)
    )
    states = jnp.concatenate([state[None, ...], states], axis=0)

    plt.figure()
    plt.title("Simulated trajectory")
    plt.plot(states[:, 0], states[:, 1], "g-")
    plt.plot(2, 2, "ro", label="Starting location")
    plt.plot(0, 0, "rx", label="Target location")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.axis("equal")
    plt.show()

    plt.figure()
    plt.plot(actions[:, 0])
    plt.plot(actions[:, 1])
    plt.xlabel("Time")
    plt.ylabel("Action")
    plt.show()
