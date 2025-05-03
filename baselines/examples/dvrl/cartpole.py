import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue

from ppomdp.smc.utils import belief_init, belief_update
from baselines.common import get_pomdp
from baselines.dvrl import (
    DVRL,
    pomdp_rollout,
    gradient_step,
    policy_evaluation,
    create_train_state,
)

import matplotlib.pyplot as plt


if __name__ == "__main__":
    config = DVRL(
        num_belief_particles=32,
        total_time_steps=100000,
        buffer_size=100000,
        learning_starts=5000,
    )

    env_obj = get_pomdp("cartpole")

    key = random.key(0)
    key, sub_key = random.split(key)
    train_state, _, _ = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=config.policy_lr,
        critic_lr=config.critic_lr,
        num_belief_particles=config.num_belief_particles,
    )

    key, init_key = random.split(key)
    pomdp_states = pomdp_rollout(
        rng_key=init_key,
        env_obj=env_obj,
        policy_state=train_state.policy_state,
        num_belief_particles=config.num_belief_particles,
        random_actions=True
    )

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], pomdp_states)
    buffer_obj = UniformSamplingQueue(
        max_replay_size=config.buffer_size,
        dummy_data_sample=buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    # Initialize the buffer state.
    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, pomdp_states)

    # Training loop.
    for global_step in range(
        env_obj.num_time_steps,
        config.total_time_steps,
        env_obj.num_time_steps
    ):
        train = global_step > config.learning_starts

        key, sub_key = random.split(key)
        pomdp_states = pomdp_rollout(
            rng_key=sub_key,
            env_obj=env_obj,
            policy_state=train_state.policy_state,
            num_belief_particles=config.num_belief_particles,
            random_actions=not train,
        )
        buffer_state = buffer_obj.insert(buffer_state, pomdp_states)

        if train:
            for _ in range(env_obj.num_time_steps):
                buffer_state, pomdp_states_batch = buffer_obj.sample(buffer_state)
                key, train_key = random.split(key)
                train_state, *_ = gradient_step(
                    rng_key=train_key,
                    train_state=train_state,
                    pomdp_state=pomdp_states_batch,
                    alpha=config.alpha,
                    gamma=config.gamma,
                    tau=config.tau,
                )

        if global_step % (20 * env_obj.num_time_steps) == 0:
            key, eval_key = random.split(key)
            avg_return, *_ = policy_evaluation(
                rng_key=eval_key,
                env_obj=env_obj,
                policy_state=train_state.policy_state,
                num_belief_particles=config.num_belief_particles
            )

            print(f"Step: {global_step:6d} | Average return: {avg_return:6.2f}")


    # Evaluate the learned policy.
    def body(carry, rng_key):
        state, belief = carry
        action_key, state_key, obs_key, pf_key = random.split(rng_key, 4)
        _, _, action = \
            train_state.policy_state.apply_fn(
                rng_key=action_key,
                particles=belief.particles,
                weights=belief.weights,
                params=train_state.policy_state.params,
            )
        state = env_obj.trans_model.sample(state_key, state, action)
        observation = env_obj.obs_model.sample(obs_key, state)
        belief = belief_update(
            rng_key=pf_key,
            trans_model=env_obj.trans_model,
            obs_model=env_obj.obs_model,
            belief_state=belief,
            observation=observation,
            action=action,
        )
        return (state, belief), (state, action)

    key, obs_key, belief_key = random.split(key, 3)
    init_state = env_obj.init_dist.mean()
    init_observation = env_obj.obs_model.sample(obs_key, init_state)

    init_belief = belief_init(
        rng_key=belief_key,
        belief_prior=env_obj.belief_prior,
        obs_model=env_obj.obs_model,
        observation=init_observation,
        num_belief_particles=config.num_belief_particles,
    )

    keys = random.split(key, env_obj.num_time_steps)
    _, (states, actions) = jax.lax.scan(
        f=body,
        init=(init_state, init_belief),
        xs=keys,
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
