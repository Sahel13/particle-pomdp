#!/usr/bin/env python3
"""
Multi-seed experiment runner for P3O algorithm.
This script uses the P3OConfig struct to run P3O experiments over multiple seeds.
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
jax.config.update("jax_enable_x64", True)

import optax

from jax import random
from jax import numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_pathwise
)
from ppomdp.utils import batch_data, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling
from ppomdp.config import P3OExperiment

from wandb_logger import WandbLogger
from common import get_pomdp, get_unique_identifier



def run_single_seed(config: P3OExperiment, seed: int) -> None:
    """Run a single seed experiment."""

    # Get environment
    env_obj = get_pomdp(config.env_id)

    # Set up wandb logging if enabled
    logger = None
    if config.use_logger:
        # Create experiment config dictionary
        p3o_config = {
            "algorithm": "p3o",
            "environment": config.env_id,
            "num_seeds": config.num_seeds,
            "cuda_device": config.cuda_device,
            "total_time_steps": config.total_time_steps,
            "num_history_particles": config.num_history_particles,
            "num_belief_particles": config.num_belief_particles,
            "slew_rate_penalty": config.slew_rate_penalty,
            "tempering": config.tempering,
            "backward_sampling": config.backward_sampling,
            "backward_sampling_mult": config.backward_sampling_mult,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "init_std": config.init_std,
        }

        experiment_name = f"{config.experiment_group}-seed-{seed}"

        # Initialize logger with specific parameters
        logger = WandbLogger(
            project_name=config.project_name,
            experiment_name=experiment_name,
            experiment_group=config.experiment_group,
            experiment_tags=config.experiment_tags,
            experiment_config=p3o_config,
            logger_directory=config.logger_directory
        )

    total_time_steps = config.total_time_steps
    num_history_particles = config.num_history_particles
    num_belief_particles = config.num_belief_particles
    slew_rate_penalty = config.slew_rate_penalty
    tempering = config.tempering
    backward_sampling = config.backward_sampling
    backward_sampling_mult = config.backward_sampling_mult
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    init_std = config.init_std

    num_target_samples = int(num_history_particles * backward_sampling_mult) \
        if backward_sampling else num_history_particles

    history_resample_fn = systematic_resampling
    belief_resample_fn = multinomial_resampling \
        if backward_sampling else systematic_resampling

    # Initialize JAX random key
    key = random.key(seed)

    # Create network and policy
    encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=(256, 256),
        recurr_sizes=(128, 128),
        use_layer_norm=True,
    )
    decoder = NeuralGaussDecoder(
        decoder_sizes=(256, 256),
        output_dim=env_obj.action_dim,
        init_log_std=constant(jnp.log(init_std)),
    )
    bijector = Block(Tanh(), ndims=1)

    policy = create_recurrent_neural_gauss_policy(
        encoder=encoder,
        decoder=decoder,
        bijector=bijector
    )
    key, sub_key = random.split(key, 2)
    params = policy.init(
        rng_key=sub_key,
        obs_dim=env_obj.obs_dim,
        action_dim=env_obj.action_dim,
        batch_dim=num_history_particles,
    )
    learner = TrainState.create(
        params=params,
        apply_fn=lambda *_: None,
        tx=optax.adam(learning_rate)
    )

    num_steps = 0

    # Check policy performance before training
    key, sub_key = random.split(key)
    rewards, *_ = policy_evaluation(
        rng_key=sub_key,
        num_time_steps=env_obj.num_time_steps,
        num_trajectory_samples=1024,
        num_belief_particles=num_belief_particles,
        init_dist=env_obj.init_dist,
        belief_prior=env_obj.belief_prior,
        policy=policy,
        policy_params=learner.params,
        trans_model=env_obj.trans_model,
        obs_model=env_obj.obs_model,
        reward_fn=env_obj.reward_fn,
        stochastic=False
    )
    avg_return = jnp.mean(jnp.sum(rewards, axis=0))

    if logger:
        logger.log_metrics({"average_return": avg_return, "step": num_steps})

    print(f"Step: {num_steps:6d} | Average return: {avg_return:8.3f}")

    # Training loop
    while num_steps <= total_time_steps:
        # Run the particle filter
        key, sub_key = random.split(key)
        history_states, belief_states, belief_infos, log_marginal = smc(
            rng_key=sub_key,
            num_time_steps=env_obj.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            belief_prior=env_obj.belief_prior,
            policy_prior=policy,
            policy_prior_params=learner.params,
            trans_model=env_obj.trans_model,
            obs_model=env_obj.obs_model,
            reward_fn=env_obj.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
            history_resample_fn=history_resample_fn,
            belief_resample_fn=belief_resample_fn,
        )
        num_steps += num_history_particles * (env_obj.num_time_steps + 1)

        if backward_sampling:
            # backward sample history states
            key, sub_key = random.split(key)
            traced_history, _ = mcmc_backward_sampling(
                rng_key=sub_key,
                num_samples=num_target_samples,
                policy_prior=policy,
                policy_prior_params=learner.params,
                trans_model=env_obj.trans_model,
                reward_fn=env_obj.reward_fn,
                slew_rate_penalty=slew_rate_penalty,
                tempering=tempering,
                history_states=history_states,
                belief_states=belief_states,
            )
        else:
            # genealogy tracking of history states
            key, sub_key = random.split(key)
            traced_history, _, _ = backward_tracing(
                sub_key, history_states, belief_states, belief_infos
            )

        # update policy parameters
        key, sub_key = random.split(key)
        batch_indices = batch_data(sub_key, num_target_samples, batch_size)
        for batch_idx in batch_indices:
            action_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.actions)
            observation_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.observations)
            learner, _ = train_recurrent_neural_gauss_policy_pathwise(
                policy=policy,
                learner=learner,
                actions=action_batch,
                observations=observation_batch,
            )

        # Evaluate the policy
        key, sub_key = random.split(key)
        rewards, *_ = policy_evaluation(
            rng_key=sub_key,
            num_time_steps=env_obj.num_time_steps,
            num_trajectory_samples=1024,
            num_belief_particles=num_belief_particles,
            init_dist=env_obj.init_dist,
            belief_prior=env_obj.belief_prior,
            policy=policy,
            policy_params=learner.params,
            trans_model=env_obj.trans_model,
            obs_model=env_obj.obs_model,
            reward_fn=env_obj.reward_fn,
            stochastic=False
        )
        avg_return = jnp.mean(jnp.sum(rewards, axis=0))

        if logger:
            logger.log_metrics({"average_return": avg_return, "step": num_steps})

        print(
            f"Step: {num_steps:6d} | "
            f"Log marginal: {log_marginal:8.3f} | "
            f"Average return: {avg_return:8.3f} | "
            f"Entropy: {policy.entropy(learner.params):8.3}"
        )

    # Finish wandb logging if enabled
    if logger:
        logger.finish()

    return None


def main(config: P3OExperiment) -> None:
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

    print(f"Experiments completed.")


if __name__ == "__main__":
    config = tyro.cli(P3OExperiment)
    main(config)
