import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from baselines.slac.slac import (
    SLACConfig,
    pomdp_init,
    pomdp_step,
    create_train_state,
    step_and_train,
)
from baselines.slac.utils import get_qmdp_state
from ppomdp.envs.pomdps import CartPolePOMDP as env_obj

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = SLACConfig()

    key = random.key(config.seed)
    key, sub_key = random.split(key)
    train_state, policy_network, _ = create_train_state(sub_key, env_obj, config.policy_lr, config.critic_lr)

    key, init_key = random.split(key)
    pomdp_state = pomdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        policy_state=train_state.policy_state,
        policy_network=policy_network,
        random_actions=True,
    )
    qmdp_state = get_qmdp_state(pomdp_state)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], qmdp_state)
    buffer_obj = UniformSamplingQueue(
        config.buffer_size,
        buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, qmdp_state)

    # Pre-fill the buffer with random actions.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        pomdp_state = pomdp_step(
            rng_key=sub_key,
            env_obj=env_obj,
            pomdp_state=pomdp_state,
            policy_state=train_state.policy_state,
            policy_network=policy_network,
            random_actions=True,
        )
        qmdp_state = get_qmdp_state(pomdp_state)
        buffer_state = buffer_obj.insert(buffer_state, qmdp_state)
        if jnp.all(pomdp_state.done_flags == 1):
            print(
                f"Step: {global_step:6d} | "
                + f"Episodic reward: {pomdp_state.total_rewards.mean():6.2f}"
            )

    # Ensure that training starts with a fresh episode.
    pomdp_state = pomdp_state._replace(done_flags=jnp.ones(env_obj.num_envs))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env_obj.num_time_steps + 1)

    # Training loop - slightly faster training with `jax.lax.scan`.
    for global_step in range(
        config.learning_starts, config.total_timesteps, steps_per_epoch
    ):
        key, sub_key = random.split(key)
        pomdp_state, buffer_state, train_state = \
            step_and_train(
                sub_key,
                env_obj,
                pomdp_state,
                buffer_obj,
                buffer_state,
                train_state,
                policy_network,
                steps_per_epoch,
                config.alpha,
                config.gamma
            )
        print(
            f"Step: {global_step + steps_per_epoch:7d} | "
            + f"Episodic reward: {pomdp_state.total_rewards.mean():10.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Evaluate the learned policy.
    key, state_key, obs_key = random.split(key, 3)
    state = env_obj.prior_dist.sample(seed=state_key)
    observation = env_obj.obs_model.sample(obs_key, state)
    lstm_carry = policy_network.reset(1)

    def body(carry, rng_key):
        _state, _lstm_carry, _observation = carry
        _action_key, _state_key, _obs_key = random.split(rng_key, 3)
        _lstm_carry, _, _, _action = train_state.policy_state.apply_fn(
            rng_key=_action_key,
            params=train_state.policy_state.params,
            carry=_lstm_carry,
            observation=_observation
        )
        _state = env_obj.trans_model.sample(_state_key, _state, _action[0])
        _observation = env_obj.obs_model.sample(_obs_key, _state)
        return (_state, _lstm_carry, _observation), (_state, _observation, _action[0])


    keys = random.split(key, env_obj.num_time_steps)
    _, (states, _, actions) = jax.lax.scan(
        body, (state, lstm_carry, observation), keys
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
