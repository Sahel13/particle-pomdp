#!/usr/bin/env python3
"""
Multi-seed experiment runner for NSMC algorithm.
This script uses the NSMCConfig struct to run NSMC experiments over multiple seeds.
"""

import os
import tyro
from tqdm import tqdm

import jax
from jax import random
from jax import numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.bijector import Tanh
from ppomdp.gauss import (
    RecurrentNeuralGauss,
    create_recurrent_gauss_policy,
    train_recurrent_gauss_policy_stepwise
)
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data, flatten_particle_trajectories, policy_evaluation
from ppomdp.config import NSMCExperiment

from wandb_logger import WandbLogger
from common import get_pomdp, get_unique_identifier


def create_network(
    encoder_size, recurr_size, decoder_size, action_dim
):
    """Create the neural network architecture."""
    encoder = GRUEncoder(
        feature_fn=lambda x: x,
        encoder_size=encoder_size,
        recurr_size=recurr_size
    )
    decoder = MLPDecoder(
        decoder_size=decoder_size,
        output_dim=action_dim,
    )
    init_log_std = jnp.ones(action_dim)
    network = RecurrentNeuralGauss(
        encoder=encoder,
        decoder=decoder,
        init_log_std=constant(init_log_std),
    )
    return network


def run_single_seed(config: NSMCExperiment, seed: int) -> None:
    """Run a single seed experiment."""

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_device)

    # Get environment
    env_obj = get_pomdp(config.env_id)

    # Set up wandb logging if enabled
    logger = None
    if config.use_logger:
        # Create experiment config dictionary
        nsmc_config = {
            "algorithm": "nsmc",
            "environment": config.env_id,
            "num_seeds": config.num_seeds,
            "cuda_device": config.cuda_device,
            "num_history_particles": config.num_history_particles,
            "num_belief_particles": config.num_belief_particles,
            "total_time_steps": config.total_time_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "tempering": config.tempering,
            "slew_rate_penalty": config.slew_rate_penalty,
            "encoder_size": config.encoder_size,
            "recurr_size": config.recurr_size,
            "decoder_size": config.decoder_size,
        }

        experiment_name = f"{config.experiment_group}-seed-{seed}"

        # Initialize logger with specific parameters
        logger = WandbLogger(
            project_name=config.project_name,
            experiment_name=experiment_name,
            experiment_group=config.experiment_group,
            experiment_tags=config.experiment_tags,
            experiment_config=nsmc_config,
            logger_directory=config.logger_directory
        )

    num_history_particles = config.num_history_particles
    num_belief_particles = config.num_belief_particles
    total_time_steps = config.total_time_steps
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    tempering = config.tempering
    slew_rate_penalty = config.slew_rate_penalty
    encoder_size = config.encoder_size
    recurr_size = config.recurr_size
    decoder_size = config.decoder_size

    # Initialize JAX random key
    key = random.key(seed)

    # Create network and policy
    bijector = Block(Tanh(), ndims=1)
    network = create_network(
        encoder_size,
        recurr_size,
        decoder_size,
        env_obj.action_dim,
    )

    key, sub_key = random.split(key)
    policy = create_recurrent_gauss_policy(network, bijector)
    train_state = policy.init(
        rng_key=sub_key,
        input_dim=env_obj.obs_dim,
        output_dim=env_obj.action_dim,
        batch_dim=num_history_particles,
        learning_rate=learning_rate,
    )

    num_steps = 0

    # Check policy performance before training
    key, sub_key = random.split(key)
    expected_reward, *_ = policy_evaluation(sub_key, env_obj, policy, train_state.params)
    print(f"Step: {num_steps:6d} | Expected reward: {expected_reward:8.3f}")

    if logger:
        logger.log_metrics({
            "expected_reward": expected_reward,
            "policy_log_std": train_state.params['log_std'][0]
        }, step=num_steps)

    # Training loop
    while num_steps <= total_time_steps:
        # Run the particle filter
        key, sub_key = random.split(key)
        history_states, belief_states, belief_infos, log_marginal = smc(
            sub_key,
            env_obj.num_time_steps,
            num_history_particles,
            num_belief_particles,
            env_obj.prior_dist,
            env_obj.trans_model,
            env_obj.obs_model,
            policy,
            train_state.params,
            env_obj.reward_fn,
            tempering,
            slew_rate_penalty,
        )
        num_steps += num_history_particles * (env_obj.num_time_steps + 1)

        # Get the smoothed trajectories using genealogy tracking
        key, sub_key = random.split(key)
        traced_history, _, _ = backward_tracing(
            sub_key, history_states, belief_states, belief_infos
        )

        # Update policy parameters
        flat_traced_history = flatten_particle_trajectories(traced_history)
        data_size = flat_traced_history.observations.shape[0]

        key, sub_key = random.split(key)
        batch_indices = batch_data(sub_key, data_size, batch_size)
        for batch_idx in batch_indices:
            history_batch = jax.tree.map(lambda x: x[batch_idx, ...], flat_traced_history)
            train_state, _ = train_recurrent_gauss_policy_stepwise(
                policy, train_state, history_batch
            )

        # Evaluate the policy
        key, sub_key = random.split(key)
        expected_reward, *_ = policy_evaluation(sub_key, env_obj, policy, train_state.params)

        if logger:
            logger.log_metrics({
                "expected_reward": expected_reward,
                "log_marginal": log_marginal,
                "policy_log_std": train_state.params['log_std'][0]
            }, step=num_steps)

        print(
            f"Step: {num_steps:6d} | "
            f"Log marginal: {log_marginal:8.3f} | "
            f"Expected reward: {expected_reward:8.3f} | "
            f"Log std: {train_state.params['log_std'][0]:8.4f}"
        )

    # Finish wandb logging if enabled
    if logger:
        logger.finish()

    return None


def main(config: NSMCExperiment) -> None:
    # Generate unique identifier for group
    identifier = get_unique_identifier()

    experiment_group = config.experiment_group + identifier
    config = config._replace(experiment_group=experiment_group)

    # Run experiments for each seed
    for seed in tqdm(range(config.num_seeds), desc="Running seeds"):
        run_single_seed(config, seed)

    print(f"Experiments completed.")


if __name__ == "__main__":
    config = tyro.cli(NSMCExperiment)
    main(config)
