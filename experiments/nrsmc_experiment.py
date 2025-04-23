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

import tyro
from tqdm import tqdm

import jax
import optax

from jax import random
from jax import numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.smc import rsmc, backward_tracing
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    initialize_multihead_recurrent_gauss_policy,
    train_multihead_recurrent_neural_gauss_policy_stepwise,
)
from ppomdp.utils import batch_data, policy_evaluation, flatten_trajectories
from ppomdp.config import NSMCExperiment

from wandb_logger import WandbLogger
from common import get_pomdp, get_unique_identifier



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
            "num_seeds": config.num_seeds,
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
            "damping": config.damping,
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
    encoder_dense_sizes = config.encoder_dense_sizes
    encoder_recurr_sizes = config.encoder_recurr_sizes
    decoder_dense_sizes = config.decoder_dense_sizes
    init_std = config.init_std
    damping = config.damping

    # Initialize JAX random key
    key = random.key(seed)

    # Create network and policy
    joint_bijector = Block(Tanh(), ndims=1)
    joint_encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=encoder_dense_sizes,
        recurr_sizes=encoder_recurr_sizes,
        use_layer_norm=True
    )
    prior_decoder = NeuralGaussDecoder(
        decoder_sizes=decoder_dense_sizes,
        output_dim=env_obj.action_dim,
        init_log_std=constant(jnp.log(init_std)),
    )
    policy_prior = create_recurrent_neural_gauss_policy(
        encoder=joint_encoder,
        decoder=prior_decoder,
        bijector=joint_bijector
    )
    posterior_decoder = NeuralGaussDecoder(
        decoder_sizes=decoder_dense_sizes,
        output_dim=env_obj.action_dim,
        init_log_std=constant(jnp.log(init_std)),
    )
    policy_posterior = create_recurrent_neural_gauss_policy(
        encoder=joint_encoder,
        decoder=posterior_decoder,
        bijector=joint_bijector
    )

    key, sub_key = random.split(key, 2)
    joint_params = initialize_multihead_recurrent_gauss_policy(
        rng_key=sub_key,
        obs_dim=env_obj.obs_dim,
        action_dim=env_obj.action_dim,
        batch_dim=num_history_particles,
        encoder=joint_encoder,
        prior_decoder=prior_decoder,
        posterior_decoder=posterior_decoder,
    )
    learner = TrainState.create(
        apply_fn=None,
        params=joint_params,
        tx=optax.adam(learning_rate)
    )

    num_steps = 0

    # Check policy performance before training
    policy_prior_params = {
        "encoder": learner.params["encoder"],
        "decoder": learner.params["prior_decoder"]
    }
    policy_posterior_params = {
        "encoder": learner.params["encoder"],
        "decoder": learner.params["posterior_decoder"]
    }

    key, sub_key = random.split(key)
    avg_reward, *_ = policy_evaluation(sub_key, env_obj, policy_posterior, policy_posterior_params)
    print(f"Step: {num_steps:6d} | Average reward: {avg_reward:8.3f}")

    if logger:
        logger.log_metrics({
            "average_reward": avg_reward,
            "policy_entropy": policy_posterior.entropy(policy_posterior_params),
        }, step=num_steps)

    # Training loop
    while num_steps <= total_time_steps:
        # Run the particle filter
        key, sub_key = random.split(key)
        history_states, belief_states, belief_infos, log_marginal = rsmc(
            rng_key=sub_key,
            num_time_steps=env_obj.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            init_prior=env_obj.prior_dist,
            policy_prior=policy_prior,
            policy_prior_params=policy_prior_params,
            policy_posterior=policy_posterior,
            policy_posterior_params=policy_posterior_params,
            trans_model=env_obj.trans_model,
            obs_model=env_obj.obs_model,
            reward_fn=env_obj.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
            damping=damping,
        )
        num_steps += num_history_particles * (env_obj.num_time_steps + 1)

        # Get the smoothed trajectories using genealogy tracking
        key, sub_key = random.split(key)
        traced_history, _, _ = backward_tracing(
            sub_key, history_states, belief_states, belief_infos
        )

        # Update policy parameters
        actions, next_actions, observations, next_observations, carry = \
            flatten_trajectories(traced_history)
        data_size, _ = observations.shape

        key, sub_key = random.split(key)
        batch_indices = batch_data(sub_key, data_size, batch_size)
        for batch_idx in batch_indices:
            action_batch = jax.tree.map(lambda x: x[batch_idx, ...], actions)
            next_action_batch = jax.tree.map(lambda x: x[batch_idx, ...], next_actions)
            observations_batch = jax.tree.map(lambda x: x[batch_idx, ...], observations)
            carry_batch = jax.tree.map(lambda x: x[batch_idx, ...], carry)

            learner, _ = train_multihead_recurrent_neural_gauss_policy_stepwise(
                learner=learner,
                policy_prior=policy_prior,
                policy_posterior=policy_posterior,
                next_actions=next_action_batch,
                carry=carry_batch,
                actions=action_batch,
                observations=observations_batch,
                damping=damping
            )

        policy_prior_params = {
            "encoder": learner.params["encoder"],
            "decoder": learner.params["prior_decoder"]
        }
        policy_posterior_params = {
            "encoder": learner.params["encoder"],
            "decoder": learner.params["posterior_decoder"]
        }

        # Evaluate the policy
        key, sub_key = random.split(key)
        avg_reward, *_ = policy_evaluation(sub_key, env_obj, policy_posterior, policy_posterior_params)

        if logger:
            logger.log_metrics({
                "average_reward": avg_reward,
                "log_marginal": log_marginal,
                "policy_entropy": policy_posterior.entropy(policy_posterior_params),
            }, step=num_steps)

        print(
            f"Step: {num_steps:6d} | "
            f"Log marginal: {log_marginal:8.3f} | "
            f"Average reward: {avg_reward:8.3f} | "
            f"Entropy: {policy_posterior.entropy(policy_posterior_params):8.4f}"
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
