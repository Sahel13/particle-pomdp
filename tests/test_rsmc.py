import pytest

import jax
from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    initialize_multihead_recurrent_gauss_policy
)
from ppomdp.smc import smc, regularized_smc

from ppomdp.envs.pomdps import LightDark1DEnv as env


def test_smc_equivalence():
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(123)

    # Common parameters
    num_history_particles = 256
    num_belief_particles = 128
    slew_rate_penalty = 0.001
    tempering = 10.0

    # Create policy
    bijector = Block(Tanh(), ndims=1)
    encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=(256, 256),
        recurr_sizes=(64, 64),
        use_layer_norm=True
    )
    decoder = NeuralGaussDecoder(
        decoder_size=(256, 256),
        output_dim=env.action_dim,
        init_log_std=constant(jnp.log(1.0)),
    )
    policy = create_recurrent_neural_gauss_policy(
        encoder=encoder,
        decoder=decoder,
        bijector=bijector
    )

    # Initialize policy parameters
    key, sub_key = random.split(rng_key)
    params = policy.init(
        rng_key=sub_key,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        batch_dim=num_history_particles,
    )

    # Split keys for both SMC runs
    key, smc_key = random.split(key, 2)

    # Run regular SMC
    history_states_smc, belief_states_smc, belief_infos_smc, log_marginal_smc = smc(
        rng_key=smc_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=env.prior_dist,
        policy_prior=policy,
        policy_prior_params=params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
    )

    # Run regularized SMC with damping=0
    history_states_rsmc, belief_states_rsmc, belief_infos_rsmc, log_marginal_rsmc = regularized_smc(
        rng_key=smc_key,  # Use the same key as regular SMC
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=env.prior_dist,
        policy_prior=policy,
        policy_prior_params=params,
        policy_posterior=policy,  # Use same policy for posterior
        policy_posterior_params=params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
        damping=0.0,  # Set damping to 0
    )

    # Compare results
    # Compare history states
    assert jnp.allclose(history_states_smc.particles.actions, history_states_rsmc.particles.actions)
    assert jnp.allclose(history_states_smc.particles.observations, history_states_rsmc.particles.observations)
    assert jnp.allclose(history_states_smc.weights, history_states_rsmc.weights)

    # Compare belief states
    assert jnp.allclose(belief_states_smc.particles, belief_states_rsmc.particles)
    assert jnp.allclose(belief_states_smc.weights, belief_states_rsmc.weights)

    # Compare belief infos
    assert jnp.allclose(belief_infos_smc.mean, belief_infos_rsmc.mean)
    assert jnp.allclose(belief_infos_smc.covar, belief_infos_rsmc.covar)

    # Compare log marginal
    assert jnp.allclose(log_marginal_smc, log_marginal_rsmc)


def test_smc_with_different_policies():
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(123)

    # Common parameters
    num_history_particles = 256
    num_belief_particles = 128
    slew_rate_penalty = 0.001
    tempering = 10.0

    # Create shared encoder
    bijector = Block(Tanh(), ndims=1)
    encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=(256, 256),
        recurr_sizes=(64, 64),
        use_layer_norm=True
    )

    # Create different decoders for prior and posterior
    prior_decoder = NeuralGaussDecoder(
        decoder_size=(256, 256),
        output_dim=env.action_dim,
        init_log_std=constant(jnp.log(1.0)),  # Standard initialization
    )
    posterior_decoder = NeuralGaussDecoder(
        decoder_size=(256, 256),
        output_dim=env.action_dim,
        init_log_std=constant(jnp.log(0.5)),  # Different initialization
    )

    # Initialize multihead policy using the initialization function
    key, sub_key = random.split(rng_key)
    params = initialize_multihead_recurrent_gauss_policy(
        rng_key=sub_key,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        batch_dim=num_history_particles,
        encoder=encoder,
        prior_decoder=prior_decoder,
        posterior_decoder=posterior_decoder,
    )

    # Create policies with the initialized parameters
    policy_prior = create_recurrent_neural_gauss_policy(
        encoder=encoder,
        decoder=prior_decoder,
        bijector=bijector
    )
    policy_posterior = create_recurrent_neural_gauss_policy(
        encoder=encoder,
        decoder=posterior_decoder,
        bijector=bijector
    )

    policy_prior_params = {"encoder": params["encoder"], "decoder": params["prior_decoder"]}
    policy_posterior_params = {"encoder": params["encoder"], "decoder": params["posterior_decoder"]}

    # Split keys for both SMC runs
    key, smc_key = random.split(key, 2)

    # Run regular SMC with prior policy
    history_states_smc, belief_states_smc, belief_infos_smc, log_marginal_smc = smc(
        rng_key=smc_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=env.prior_dist,
        policy_prior=policy_prior,
        policy_prior_params=policy_prior_params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
    )

    # Run regularized SMC with both policies
    history_states_rsmc, belief_states_rsmc, belief_infos_rsmc, log_marginal_rsmc = regularized_smc(
        rng_key=smc_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=env.prior_dist,
        policy_prior=policy_prior,
        policy_prior_params=policy_prior_params,
        policy_posterior=policy_posterior,
        policy_posterior_params=policy_posterior_params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
        damping=0.0,  # Set damping to 0
    )

    # Compare results
    # Compare history states
    assert jnp.allclose(history_states_smc.particles.actions, history_states_rsmc.particles.actions)
    assert jnp.allclose(history_states_smc.particles.observations, history_states_rsmc.particles.observations)
    assert jnp.allclose(history_states_smc.weights, history_states_rsmc.weights)

    # Compare belief states
    assert jnp.allclose(belief_states_smc.particles, belief_states_rsmc.particles)
    assert jnp.allclose(belief_states_smc.weights, belief_states_rsmc.weights)

    # Compare belief infos
    assert jnp.allclose(belief_infos_smc.mean, belief_infos_rsmc.mean)
    assert jnp.allclose(belief_infos_smc.covar, belief_infos_rsmc.covar)

    # Compare log marginal
    assert jnp.allclose(log_marginal_smc, log_marginal_rsmc)


def test_smc_gradient_equivalence():
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(123)

    # Common parameters
    num_history_particles = 256
    num_belief_particles = 128
    slew_rate_penalty = 0.001
    tempering = 10.0

    # Create shared encoder
    bijector = Block(Tanh(), ndims=1)
    encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=(256, 256),
        recurr_sizes=(64, 64),
        use_layer_norm=True
    )

    # Create different decoders for prior and posterior
    prior_decoder = NeuralGaussDecoder(
        decoder_size=(256, 256),
        output_dim=env.action_dim,
        init_log_std=constant(jnp.log(1.0)),  # Standard initialization
    )
    posterior_decoder = NeuralGaussDecoder(
        decoder_size=(256, 256),
        output_dim=env.action_dim,
        init_log_std=constant(jnp.log(0.5)),  # Different initialization
    )

    # Initialize multihead policy using the initialization function
    key, sub_key = random.split(rng_key)
    joint_params = initialize_multihead_recurrent_gauss_policy(
        rng_key=sub_key,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        batch_dim=num_history_particles,
        encoder=encoder,
        prior_decoder=prior_decoder,
        posterior_decoder=posterior_decoder,
    )

    # Create policies with the initialized parameters
    policy_prior = create_recurrent_neural_gauss_policy(
        encoder=encoder,
        decoder=prior_decoder,
        bijector=bijector
    )
    policy_posterior = create_recurrent_neural_gauss_policy(
        encoder=encoder,
        decoder=posterior_decoder,
        bijector=bijector
    )

    # Create parameter structures for both cases
    policy_prior_params = {"encoder": joint_params["encoder"], "decoder": joint_params["prior_decoder"]}
    policy_posterior_params = {"encoder": joint_params["encoder"], "decoder": joint_params["posterior_decoder"]}

    # Create separate learners for single and multihead policies
    from flax.training.train_state import TrainState
    from optax import adam
    learner_single = TrainState.create(
        apply_fn=None,
        params=policy_prior_params,
        tx=adam(1e-4)
    )
    learner_multihead = TrainState.create(
        apply_fn=None,
        params=joint_params,
        tx=adam(1e-4)
    )

    # Run SMC to get particles
    key, smc_key = random.split(key)
    history_states, belief_states, belief_infos, _ = smc(
        rng_key=smc_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=env.prior_dist,
        policy_prior=policy_prior,
        policy_prior_params=policy_prior_params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
    )

    # Get particles for training
    particles = history_states.particles

    # Train with regular SMC (using prior policy)
    from ppomdp.policy.gauss import train_recurrent_neural_gauss_policy_pathwise
    learner_single, loss_smc = train_recurrent_neural_gauss_policy_pathwise(
        learner=learner_single,
        policy=policy_prior,
        particles=particles
    )

    # Train with multihead (using both prior and posterior policies)
    from ppomdp.policy.gauss import train_multihead_recurrent_neural_gauss_policy_pathwise
    learner_multihead, loss_multihead = train_multihead_recurrent_neural_gauss_policy_pathwise(
        learner=learner_multihead,
        policy_prior=policy_prior,
        policy_posterior=policy_posterior,
        particles=particles,
        damping=0.0
    )

    # Compare parameter norms
    def get_norm(params):
        return jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)))

    single_encoder_norm = get_norm(learner_single.params["encoder"])
    single_decoder_norm = get_norm(learner_single.params["decoder"])
    multihead_encoder_norm = get_norm(learner_multihead.params["encoder"])
    multihead_prior_decoder_norm = get_norm(learner_multihead.params["prior_decoder"])

    assert jnp.allclose(single_encoder_norm, multihead_encoder_norm, rtol=1e-4, atol=1e-4)
    assert jnp.allclose(single_decoder_norm, multihead_prior_decoder_norm, rtol=1e-4, atol=1e-4)

    # Compare losses
    assert jnp.allclose(loss_smc, loss_multihead, rtol=1e-4, atol=1e-4)