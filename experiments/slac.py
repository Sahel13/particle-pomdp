import csv
import os
from typing import NamedTuple

import common
import jax
from brax.training.replay_buffers import UniformSamplingQueue
from jax import random

from baselines.slac.slac import (
    create_train_state,
    evaluate,
    pomdp_init,
    pomdp_step,
    step_and_train,
)


class SLACConfig(NamedTuple):
    num_belief_particles: int = 64
    total_timesteps: int = int(5e5)
    buffer_size: int = int(5e5)
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 1e-4
    critic_lr: float = 1e-3
    alpha: float = 0.2
    gamma: float = 0.995
    tau: float = 0.005


config = SLACConfig()
cmd_args = common.get_cmd_args()
env = common.get_env(cmd_args.env)
key = random.key(cmd_args.seed)

# Set up logging.
file_name = f"training_log_seed_{cmd_args.seed}.csv"
file_path = os.path.join(cmd_args.log_dir, file_name)
logger = [["Step", "Episodic reward"]]

key, sub_key = random.split(key)
train_state, policy_network, _ = create_train_state(sub_key, env, config)

key, init_key = random.split(key)
pomdp_state = pomdp_init(
    rng_key=init_key,
    env_obj=env,
    alg_cfg=config,
    policy_state=train_state.policy_state,
    policy_network=policy_network,
    random_actions=True,
)

# Set up the replay buffer from Brax.
buffer_entry_prototype = jax.tree.map(lambda x: x[0], pomdp_state)
buffer_obj = UniformSamplingQueue(
    config.buffer_size, buffer_entry_prototype, sample_batch_size=config.batch_size
)

buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

key, buffer_key = random.split(key)
buffer_state = buffer_obj.init(buffer_key)
buffer_state = buffer_obj.insert(buffer_state, pomdp_state)

# Pre-fill the buffer with random actions.
for global_step in range(1, config.learning_starts):
    key, sub_key = random.split(key)
    pomdp_state = pomdp_step(
        rng_key=sub_key,
        env_obj=env,
        alg_cfg=config,
        pomdp_state=pomdp_state,
        policy_state=train_state.policy_state,
        policy_network=policy_network,
        random_actions=True,
    )
    buffer_state = buffer_obj.insert(buffer_state, pomdp_state)

    if global_step % 2000 == 0:
        key, sub_key = random.split(key)
        expected_reward, *_ = evaluate(sub_key, env, train_state, policy_network)
        logger.append([global_step, expected_reward])
        print(f"Step: {global_step:6d} | Episodic reward: {expected_reward:6.2f}")

# Number of steps to take using the `lax.scan` loop (and how often to evaluate the policy).
steps_per_epoch = 2000

# Training loop.
for global_step in range(
    config.learning_starts, config.total_timesteps, steps_per_epoch
):
    key, sub_key = random.split(key)
    pomdp_state, buffer_state, train_state = step_and_train(
        sub_key,
        env,
        config,
        pomdp_state,
        buffer_obj,
        buffer_state,
        train_state,
        policy_network,
        steps_per_epoch,
    )
    key, sub_key = random.split(key)
    expected_reward, states, actions = evaluate(sub_key, env, train_state, policy_network)
    logger.append([global_step + steps_per_epoch, expected_reward])
    print(
        f"Step: {global_step + steps_per_epoch:7d} | "
        + f"Expected reward: {expected_reward:10.2f} | "
        + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
    )

# Save the training data.
with open(file_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(logger)
