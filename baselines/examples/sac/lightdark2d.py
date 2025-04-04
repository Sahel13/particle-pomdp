import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from baselines.sac.sac import (
    SACConfig,
    mdp_init,
    mdp_step,
    create_train_state,
    step_and_train,
)
from ppomdp.envs.mdps import LightDark2DEnv as env_obj

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = SACConfig()

    key = random.key(config.seed)
    key, sub_key = random.split(key)
    train_state = create_train_state(sub_key, env_obj, config)

    key, sub_key = random.split(key)
    mdp_state = mdp_init(
        rng_key=sub_key,
        env_obj=env_obj,
        alg_config=config,
        policy_state=train_state.policy_state,
        random_actions=True
    )

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], mdp_state)
    buffer_obj = UniformSamplingQueue(
        config.buffer_size,
        buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    key, sub_key = random.split(key)
    buffer_state = buffer_obj.init(sub_key)
    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_state = buffer_obj.insert(buffer_state, mdp_state)

    # Pre-populate the buffer with random trajectories.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        mdp_state = mdp_step(
            rng_key=sub_key,
            env_obj=env_obj,
            alg_config=config,
            mdp_state=mdp_state,
            policy_state=train_state.policy_state,
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
                sub_key,
                env_obj,
                config,
                mdp_state,
                buffer_obj,
                buffer_state,
                train_state,
                steps_per_epoch,
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
            params=train_state.policy_state.params,
            state=_state,
            time_idx=_time_idx
        )
        _state = env_obj.trans_model.sample(_state_key, _state, _action)
        return (_state, _time_idx + 1), (_state, _action)


    _, (states, actions) = jax.lax.scan(
        body_fn, (state, 0), random.split(key, env_obj.num_time_steps)
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
