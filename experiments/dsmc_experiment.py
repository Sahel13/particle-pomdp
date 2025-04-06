#!/usr/bin/env python3
"""
Multi-seed experiment runner for DSMC algorithm.
This script uses the DSMCExperiment struct to run DSMC experiments over multiple seeds.
"""

import os
import time
import uuid
import tyro
from tqdm import tqdm

from baselines.dsmc import DSMCExperiment
from baselines.dsmc import (
    pomdp_init,
    pomdp_step,
    create_train_state,
    step_and_train,
    policy_evaluate,
)
from wandb_logger import WandbLogger
from common import get_env


def generate_experiment_name(config: DSMCExperiment) -> str:
    """Generate a unique experiment name."""
    if config.experiment_name:
        return config.experiment_name
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"dsmc-{config.env_id}-{timestamp}-{unique_id}"


def run_single_seed(config: DSMCExperiment, seed: int) -> None:
    """Run a single seed experiment."""
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_device)
    
    # Get environment
    env_obj = get_env(config.env_id)

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
        
        # Initialize logger with specific parameters
        logger = WandbLogger(
            project_name=config.project_name,
            experiment_name=config.experiment_name,
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
    import jax
    from jax import random
    from jax import numpy as jnp
    key = random.key(seed)
    
    # Create train state and networks
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
            expected_reward, _, _ = policy_evaluate(
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
        expected_reward, _, _ = policy_evaluate(
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
    # Generate experiment name if not provided
    if not config.experiment_name:
        config.experiment_name = generate_experiment_name(config)
        
    # Run experiments for each seed
    for seed in tqdm(range(config.num_seeds), desc="Running seeds"):
        run_single_seed(config, seed)
        
    print(f"Experiment completed.")


if __name__ == "__main__":
    config = tyro.cli(DSMCExperiment)
    main(config)
