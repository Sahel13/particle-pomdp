#!/usr/bin/env python3
"""
Multi-seed experiment runner for NSMC algorithm.
This script uses the NSMCConfig struct to run NSMC experiments over multiple seeds.
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
import optax
import tyro
from common import get_pomdp, get_unique_identifier
from distrax import Block
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax import random
from tqdm import tqdm
from wandb_logger import WandbLogger

from ppomdp.bijector import Tanh
from ppomdp.config import NSMCExperiment
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_pathwise,
)
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data, policy_evaluation


def run_single_seed(config: NSMCExperiment, seed: int) -> None:
    """Run a single seed experiment."""

    # Get environment
    env_obj = get_pomdp(config.env_id)

    # Set up wandb logging if enabled
    logger = None
    if config.use_logger:
        # Create experiment config dictionary
        nsmc_config = {
            "algorithm": "nsmc",
            "environment": config.env_id,
            "cuda_device": config.cuda_device,
            "num_history_particles": config.num_history_particles,
            "num_belief_particles": config.num_belief_particles,
            "total_time_steps": config.total_time_steps,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "tempering": config.tempering,
            "slew_rate_penalty": config.slew_rate_penalty,
            "encoder_dense_sizes": config.encoder_dense_sizes,
            "encoder_recurr_sizes": config.encoder_recurr_sizes,
            "decoder_dense_sizes": config.decoder_dense_sizes,
        }

        experiment_name = f"{config.experiment_group}-seed-{seed}"

        # Initialize logger with specific parameters
        logger = WandbLogger(
            project_name=config.project_name,
            experiment_name=experiment_name,
            experiment_group=config.experiment_group,
            experiment_tags=config.experiment_tags,
            experiment_config=nsmc_config,
            logger_directory=config.logger_directory,
        )

    num_history_particles = config.num_history_particles
    num_belief_particles = config.num_belief_particles
    total_time_steps = config.total_time_steps
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    tempering = config.tempering
    slew_rate_penalty = config.slew_rate_penalty
    encoder_dense_sizes = config.encoder_dense_sizes
    encoder_recurr_sizes = config.encoder_recurr_sizes
    decoder_dense_sizes = config.decoder_dense_sizes
    init_std = config.init_std

    # Initialize JAX random key
    key = random.key(seed)

    # Create network and policy
    encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=encoder_dense_sizes,
        recurr_sizes=encoder_recurr_sizes,
        use_layer_norm=True,
    )
    decoder = NeuralGaussDecoder(
        decoder_sizes=decoder_dense_sizes,
        output_dim=env_obj.action_dim,
        init_log_std=constant(jnp.log(init_std)),
    )
    bijector = Block(Tanh(), ndims=1)

    policy = create_recurrent_neural_gauss_policy(
        encoder=encoder, decoder=decoder, bijector=bijector
    )
    key, sub_key = random.split(key, 2)
    params = policy.init(
        rng_key=sub_key,
        obs_dim=env_obj.obs_dim,
        action_dim=env_obj.action_dim,
        batch_dim=num_history_particles,
    )
    learner = TrainState.create(
        params=params, apply_fn=lambda *_: None, tx=optax.adam(learning_rate)
    )

    num_steps = 0

    # Check policy performance before training
    key, sub_key = random.split(key)
    expected_reward, *_ = policy_evaluation(sub_key, env_obj, policy, learner.params)
    print(f"Step: {num_steps:6d} | Expected reward: {expected_reward:8.3f}")

    if logger:
        logger.log_metrics(
            {
                "expected_reward": expected_reward,
                "policy_entropy": policy.entropy(learner.params),
            },
            step=num_steps,
        )

    # Training loop
    while num_steps <= total_time_steps:
        # Run the particle filter
        key, sub_key = random.split(key)
        history_states, belief_states, belief_infos, log_marginal = smc(
            rng_key=sub_key,
            num_time_steps=env_obj.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            init_prior=env_obj.prior_dist,
            policy_prior=policy,
            policy_prior_params=learner.params,
            trans_model=env_obj.trans_model,
            obs_model=env_obj.obs_model,
            reward_fn=env_obj.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
        )
        num_steps += num_history_particles * (env_obj.num_time_steps + 1)

        # Get the smoothed trajectories using genealogy tracking
        key, sub_key = random.split(key)
        traced_history, *_ = backward_tracing(
            sub_key, history_states, belief_states, belief_infos
        )

        # Update policy parameters
        key, batch_key = random.split(key)
        batch_indices = batch_data(batch_key, num_history_particles, batch_size)
        for batch_idx in batch_indices:
            history_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history)
            learner, _ = train_recurrent_neural_gauss_policy_pathwise(
                learner, policy, history_batch.actions, history_batch.observations
            )

        # Evaluate the policy
        key, eval_key = random.split(key)
        expected_reward, *_ = policy_evaluation(eval_key, env_obj, policy, learner.params)

        if logger:
            logger.log_metrics(
                {
                    "expected_reward": expected_reward,
                    "log_marginal": log_marginal,
                    "policy_entropy": policy.entropy(learner.params),
                },
                step=num_steps,
            )

        print(
            f"Step: {num_steps:6d} | "
            f"Log marginal: {log_marginal:8.3f} | "
            f"Expected reward: {expected_reward:8.3f}"
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
