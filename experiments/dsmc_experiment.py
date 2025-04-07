#!/usr/bin/env python3
"""
Multi-seed experiment runner for DSMC algorithm.
This script uses the DSMCExperiment struct to run DSMC experiments over multiple seeds.
"""

import os
import sys

# Set CUDA device before importing any GPU-related libraries
if len(sys.argv) > 1:
    # Parse command line arguments manually to get cuda_device
    for i, arg in enumerate(sys.argv):
        if arg == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            print(f"Setting CUDA_VISIBLE_DEVICES to {sys.argv[i + 1]}")
            break

import tyro
from tqdm import tqdm

import jax
from jax import random
from jax import numpy as jnp

from baselines.dsmc import DSMCExperiment
from baselines.dsmc import (
    pomdp_init,
    pomdp_step,
    create_train_state,
    step_and_train,
    policy_evaluation,
)
from wandb_logger import WandbLogger
from common import get_pomdp, get_unique_identifier



def run_single_seed(config: DSMCExperiment, seed: int) -> None:
    """Run a single seed experiment."""

    # Get environment
    env_obj = get_pomdp(config.env_id)

    # Set up wandb logging if enabled
    logger = None
    if config.use_logger:
        # Create experiment config dictionary
        dsmc_config = {
            "algorithm": "dsmc",
            "environment": config.env_id,
            "num_seeds": config.num_seeds,
            "cuda_device": config.cuda_device,
            "num_planner_steps": config.num_planner_steps,
            "num_planner_particles": config.num_planner_particles,
            "num_belief_particles": config.num_belief_particles,
            "total_time_steps": config.total_time_steps,
            "buffer_size": config.buffer_size,
            "learning_starts": config.learning_starts,
            "policy_lr": config.policy_lr,
            "critic_lr": config.critic_lr,
            "batch_size": config.batch_size,
            "alpha": config.alpha,
            "gamma": config.gamma,
            "tau": config.tau,
        }

        experiment_name = f"{config.experiment_group}-seed-{seed}"

        # Initialize logger with specific parameters
        logger = WandbLogger(
            project_name=config.project_name,
            experiment_name=experiment_name,
            experiment_group=config.experiment_group,
            experiment_tags=config.experiment_tags,
            experiment_config=dsmc_config,
            logger_directory=config.logger_directory
        )

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

    # Initialize JAX random key
    key = random.key(seed)
    key, sub_key = random.split(key)
    train_state, _, _ = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=policy_lr,
        critic_lr=critic_lr,
        num_planner_particles=num_planner_particles,
    )

    # Initialize POMDP state
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

    # Set up the replay buffer
    from brax.training.replay_buffers import UniformSamplingQueue
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

    # Pre-fill the buffer with random actions
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

        if global_step % 2000 == 0:
            key, sub_key = random.split(key)
            expected_reward, _, _ = policy_evaluation(
                rng_key=sub_key,
                env_obj=env_obj,
                policy_state=train_state.policy_state,
                num_belief_particles=num_belief_particles,
            )

            if logger:
                logger.log_metrics({
                    "expected_reward": expected_reward,
                    "policy_log_std": train_state.policy_state.params['log_std'][0]
                }, step=global_step)

            print(f"Step: {global_step:6d} | Expected reward: {expected_reward:6.2f}")

    # Ensure that training starts with a fresh episode
    pomdp_state = pomdp_state._replace(done_flags=jnp.ones(env_obj.num_envs, dtype=jnp.int32))

    # Number of steps to take using the `lax.scan` loop
    steps_per_epoch = 2000

    # Training loop
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

        key, sub_key = random.split(key)
        expected_reward, _, _ = policy_evaluation(
            rng_key=sub_key,
            env_obj=env_obj,
            policy_state=train_state.policy_state,
            num_belief_particles=num_belief_particles,
        )

        if logger:
            logger.log_metrics({
                "expected_reward": expected_reward,
                "policy_log_std": train_state.policy_state.params['log_std'][0]
            }, step=global_step + steps_per_epoch)

        print(
            f"Step: {global_step + steps_per_epoch:6d} | "
            + f"Expected reward: {expected_reward:6.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Finish wandb logging if enabled
    if logger:
        logger.finish()

    return None


def main(config: DSMCExperiment) -> None:
    # Generate unique identifier for group
    identifier = get_unique_identifier()

    experiment_group = config.experiment_group + identifier
    config = config._replace(experiment_group=experiment_group)

    # Run experiments for each seed
    for seed in tqdm(range(config.num_seeds), desc="Running seeds"):
        run_single_seed(config, seed)

    print(f"Experiment completed.")


if __name__ == "__main__":
    config = tyro.cli(DSMCExperiment)
    main(config)
