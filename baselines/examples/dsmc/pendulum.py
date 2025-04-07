import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from baselines.common import get_pomdp, belief_init, belief_update
from baselines.dsmc import (
    DSMC,
    pomdp_init,
    pomdp_step,
    create_train_state,
    step_and_train,
)

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = DSMC(
        num_belief_particles=32,
        total_time_steps=25000,
        buffer_size=25000,
        learning_starts=5000,
    )

    env_obj = get_pomdp("pendulum")

    num_planner_steps = config.num_planner_steps
    num_planner_particles = config.num_planner_particles
    num_belief_particles = config.num_belief_particles
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
    train_state, _, _ = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=policy_lr,
        critic_lr=critic_lr,
        num_planner_particles=num_planner_particles,
    )

    key, init_key = random.split(key)
    pomdp_state = pomdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        train_state=train_state,
        num_belief_particles=num_belief_particles,
        num_planner_particles=num_planner_particles,
        num_planner_steps=num_planner_steps,
        alpha=alpha,
        gamma=gamma,
        random_actions=True,
    )

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], pomdp_state)
    buffer_obj = UniformSamplingQueue(
        max_replay_size=buffer_size,
        dummy_data_sample=buffer_entry_prototype,
        sample_batch_size=batch_size
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, pomdp_state)

    # Pre-fill the buffer with random actions.
    for global_step in range(1, learning_starts):
        key, sub_key = random.split(key)
        pomdp_state = pomdp_step(
            rng_key=sub_key,
            env_obj=env_obj,
            train_state=train_state,
            pomdp_state=pomdp_state,
            num_belief_particles=num_belief_particles,
            num_planner_particles=num_planner_particles,
            num_planner_steps=num_planner_steps,
            alpha=alpha,
            gamma=gamma,
            random_actions=True,
        )
        buffer_state = buffer_obj.insert(buffer_state, pomdp_state)
        if jnp.all(pomdp_state.done_flags == 1):
            print(
                f"Step: {global_step:7d} | "
                + f"Episodic reward: {pomdp_state.total_rewards.mean():6.2f}"
            )

    # Ensure that training starts with a fresh episode.
    pomdp_state = pomdp_state._replace(done_flags=jnp.ones(env_obj.num_envs, dtype=jnp.int32))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env_obj.num_time_steps + 1)

    # Training loop - slightly faster training with `jax.lax.scan`.
    for global_step in range(
        learning_starts, total_time_steps, steps_per_epoch
    ):
        key, sub_key = random.split(key)
        pomdp_state, buffer_state, train_state = \
            step_and_train(
                rng_key=sub_key,
                env_obj=env_obj,
                train_state=train_state,
                pomdp_state=pomdp_state,
                buffer_obj=buffer_obj,
                buffer_state=buffer_state,
                num_belief_particles=num_belief_particles,
                num_planner_particles=num_planner_particles,
                num_planner_steps=num_planner_steps,
                num_steps=steps_per_epoch,
                alpha=alpha,
                gamma=gamma,
                tau=tau,
            )
        print(
            f"Step: {global_step + steps_per_epoch:7d} | "
            + f"Episodic reward: {pomdp_state.total_rewards.mean():6.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Evaluate the learned policy.
    key, obs_key, belief_key = random.split(key, 3)
    state = env_obj.prior_dist.mean()
    observation = env_obj.obs_model.sample(obs_key, state)
    belief_state = belief_init(belief_key, env_obj, observation, num_belief_particles)

    def body(carry, rng_key):
        _state, _belief_state = carry
        _action_key, _state_key, _obs_key, _pf_key = random.split(rng_key, 4)
        _, _, _action = \
            train_state.policy_state.apply_fn(
                rng_key=_action_key,
                particles=_belief_state.particles,
                weights=_belief_state.weights,
                params=train_state.policy_state.params,
            )
        _state = env_obj.trans_model.sample(_state_key, _state, _action)
        _observation = env_obj.obs_model.sample(_obs_key, _state)
        _belief_state = belief_update(_pf_key, env_obj, _belief_state, _observation, _action)
        return (_state, _belief_state), (_state, _action)

    keys = random.split(key, env_obj.num_time_steps)
    _, (states, actions) = jax.lax.scan(
        body, (state, belief_state), keys
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
