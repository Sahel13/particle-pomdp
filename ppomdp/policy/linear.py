from functools import partial

import jax
from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import MultivariateNormalDiag

from ppomdp.core import PRNGKey, Parameters, LinearPolicy
from ppomdp.policy.arch import LinearGaussDecoder


def create_linear_policy(
    decoder: LinearGaussDecoder,
) -> LinearPolicy:
    """Creates a stochastic linear policy that acts on weighted particle means.
    
    This policy computes the weighted mean of the particle set and applies a linear
    Gaussian decoder to produce stochastic actions. No encoder or bijector is needed.
    
    Args:
        decoder (LinearGaussDecoder): The linear Gaussian decoder network
        
    Returns:
        LinearPolicy: A policy object implementing the LinearPolicy interface
            that produces Gaussian-distributed actions based on weighted particle means
    """
    
    def sample(
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array]:
        # Compute weighted mean of particles
        weights_normalized = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)
        weighted_mean = jnp.sum(particles * weights_normalized[..., None], axis=1)
        
        # Get Gaussian parameters from decoder
        mean, log_std = decoder.apply({"params": params["decoder"]}, weighted_mean)
        
        # Create Gaussian distribution and sample
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        actions = dist.sample(seed=rng_key)
        
        return actions, mean
    
    def log_prob(
        actions: Array,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> Array:
        # Compute weighted mean of particles
        weights_normalized = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)
        weighted_mean = jnp.sum(particles * weights_normalized[..., None], axis=1)
        
        # Get Gaussian parameters from decoder
        mean, log_std = decoder.apply({"params": params["decoder"]}, weighted_mean)
        
        # Create Gaussian distribution and compute log prob
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return dist.log_prob(actions)
    
    def sample_and_log_prob(
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array, Array]:
        # Compute weighted mean of particles
        weights_normalized = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-8)
        weighted_mean = jnp.sum(particles * weights_normalized[..., None], axis=1)
        
        # Get Gaussian parameters from decoder
        mean, log_std = decoder.apply({"params": params["decoder"]}, weighted_mean)
        
        # Create Gaussian distribution and sample with log prob
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        actions, log_prob = dist.sample_and_log_prob(seed=rng_key)
        
        return actions, log_prob, mean
    
    def entropy(params: Parameters) -> Array:
        # Gaussian entropy from decoder parameters
        log_std = params["decoder"]["log_std"]
        return decoder.entropy(log_std)
    
    def reset(batch_size: int) -> None:
        return None
    
    def init(
        rng_key: PRNGKey,
        particle_dim: int,
        action_dim: int,
        batch_size: int,
        num_particles: int,
    ) -> Parameters:
        # Create dummy weighted mean to initialize decoder
        dummy_key = rng_key
        dummy_particles = random.normal(dummy_key, (batch_size, num_particles, particle_dim))
        dummy_weights = jnp.ones((batch_size, num_particles)) / num_particles
        
        # Compute dummy weighted mean
        weighted_mean = jnp.sum(dummy_particles * dummy_weights[..., None], axis=1)
        
        # Initialize decoder parameters
        decoder_params = decoder.init(rng_key, weighted_mean)["params"]
        
        return {"decoder": decoder_params}
    
    return LinearPolicy(
        dim=decoder.dim,
        init=init,
        reset=reset,
        sample=sample,
        log_prob=log_prob,
        sample_and_log_prob=sample_and_log_prob,
        entropy=entropy,
    )


@partial(jax.jit, static_argnames="policy")
def train_linear_policy(
    learner: TrainState,
    policy: LinearPolicy,
    actions: Array,
    particles: Array,
    weights: Array,
) -> tuple[TrainState, Array]:
    """Trains a linear policy using gradient-based optimization.
    
    For a stochastic linear policy, we maximize the log-likelihood of the
    target actions under the Gaussian distribution parameterized by the policy.
    
    Args:
        learner (TrainState): The current state of the training process
        policy (LinearPolicy): The linear policy to be trained
        actions (Array): The target actions
        particles (Array): The array of particle sets
        weights (Array): The array of particle weights
        
    Returns:
        tuple[TrainState, Array]: Updated learner state and computed loss
    """
    def loss_fn(params):
        # Compute negative log-likelihood
        log_probs = policy.log_prob(actions, particles, weights, params)
        return -jnp.mean(log_probs)
    
    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss 