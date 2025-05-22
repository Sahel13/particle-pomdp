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

import jax
import tyro
from tqdm import tqdm

from jax import random, numpy as jnp
from brax.training.replay_buffers import UniformSamplingQueue
from common import get_pomdp, get_unique_identifier
from wandb_logger import WandbLogger

from baselines.dsmc import (
    DSMCExperiment,
    create_train_state,
    gradient_step,
    policy_evaluation,
    pomdp_rollout,
)


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
            "num_batches": config.num_batches,
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
            logger_directory=config.logger_directory,
        )

    # Initialize JAX random key
    key = random.key(seed)
    key, sub_key = random.split(key)
    train_state, _, _ = create_train_state(
        rng_key=sub_key,
        env_obj=env_obj,
        policy_lr=config.policy_lr,
        critic_lr=config.critic_lr,
        num_planner_particles=config.num_planner_particles,
    )

    # Check policy performance before training
    key, eval_key = random.split(key)
    rewards, *_ = policy_evaluation(
        rng_key=eval_key,
        num_time_steps=env_obj.num_time_steps,
        num_trajectory_samples=1024,
        num_belief_particles=config.num_belief_particles,
        init_dist=env_obj.init_dist,
        belief_prior=env_obj.belief_prior,
        policy_state=train_state.policy_state,
        trans_model=env_obj.trans_model,
        obs_model=env_obj.obs_model,
        reward_fn=env_obj.reward_fn,
    )
    avg_return = jnp.mean(jnp.sum(rewards, axis=0))

    if logger:
        logger.log_metrics({"average_return": avg_return, "step": 0})

    print(f"Step: {0:6d} | Average return: {avg_return:6.3f}")

    # Initialize POMDP state
    key, init_key = random.split(key)
    pomdp_states = pomdp_rollout(
        rng_key=init_key,
        env_obj=env_obj,
        train_state=train_state,
        num_belief_particles=config.num_belief_particles,
        num_planner_particles=config.num_planner_particles,
        num_planner_steps=config.num_planner_steps,
        alpha=config.alpha,
        gamma=config.gamma,
        random_actions=True,
    )

    # Set up the replay buffer
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], pomdp_states)
    buffer_obj = UniformSamplingQueue(
        max_replay_size=config.buffer_size,
        dummy_data_sample=buffer_entry_prototype,
        sample_batch_size=config.batch_size,
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, pomdp_states)

    # Training loop
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
            train_state=train_state,
            num_belief_particles=config.num_belief_particles,
            num_planner_particles=config.num_planner_particles,
            num_planner_steps=config.num_planner_steps,
            alpha=config.alpha,
            gamma=config.gamma,
            random_actions=not train,
        )
        buffer_state = buffer_obj.insert(buffer_state, pomdp_states)

        if train:
            for _ in range(config.num_batches):
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
            rewards, *_ = policy_evaluation(
                rng_key=eval_key,
                num_time_steps=env_obj.num_time_steps,
                num_trajectory_samples=1024,
                num_belief_particles=config.num_belief_particles,
                init_dist=env_obj.init_dist,
                belief_prior=env_obj.belief_prior,
                policy_state=train_state.policy_state,
                trans_model=env_obj.trans_model,
                obs_model=env_obj.obs_model,
                reward_fn=env_obj.reward_fn,
            )
            avg_return = jnp.mean(jnp.sum(rewards, axis=0))

            if logger:
                logger.log_metrics({"average_return": avg_return, "step": global_step})

            print(f"Step: {global_step:6d} | Average return: {avg_return:6.2f}")

    # Finish wandb logging if enabled
    if logger:
        logger.finish()

    return None


def main(config: DSMCExperiment) -> None:
    # Generate unique identifier for group
    if config.experiment_id:
        identifier = config.experiment_id
    else:
        identifier = get_unique_identifier()

    experiment_group = config.experiment_group + identifier
    config = config._replace(experiment_group=experiment_group)

    # Run experiments for each seed
    for seed in tqdm(
        range(config.starting_seed, config.starting_seed + config.num_seeds),
        desc="Running seeds",
    ):
        run_single_seed(config, seed)

    print("Experiment completed.")


if __name__ == "__main__":
    config = tyro.cli(DSMCExperiment)
    main(config)
