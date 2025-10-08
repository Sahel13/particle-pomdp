from functools import partial

import jax
from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import (
    Bijector,
    MultivariateNormalDiag,
    Transformed,
)

from ppomdp.core import PRNGKey, Parameters, AttentionPolicy
from ppomdp.policy.arch import AttentionEncoder, NeuralGaussDecoder


def create_attention_policy(
    encoder: AttentionEncoder,
    decoder: NeuralGaussDecoder,
    bijector: Bijector
) -> AttentionPolicy:
    """Creates a policy that processes particle sets using an attention architecture.

    Args:
        encoder (AttentionEncoder): The attention encoder network
        decoder (NeuralGaussDecoder): The neural network used for the policy
        bijector (Bijector): Policy bijector to enforce action limits

    Returns:
        AttentionPolicy: A policy object implementing the AttentionPolicy interface with
            methods for sampling actions and computing probabilities
    """

    def sample(
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array]:
        encoding = encoder.apply({"params": params["encoder"]}, particles, weights)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encoding)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        actions = dist.sample(seed=rng_key)
        return actions, bijector.forward(mean)

    def log_prob(
        actions: Array,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> Array:
        encoding = encoder.apply({"params": params["encoder"]}, particles, weights)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encoding)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return dist.log_prob(actions)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        particles: Array,
        weights: Array,
        params: Parameters,
    ) -> tuple[Array, Array, Array]:
        encoding = encoder.apply({"params": params["encoder"]}, particles, weights)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encoding)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        actions, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return actions, log_prob, bijector.forward(mean)

    def entropy(params: Parameters) -> Array:
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
        dummy_key, encoder_key, decoder_key = random.split(rng_key, 3)

        # Initialize encoder network
        dummy_particles = random.normal(dummy_key, (batch_size, num_particles, particle_dim))
        dummy_weights = jnp.ones((batch_size, num_particles)) / num_particles
        encoder_params = encoder.init(encoder_key, dummy_particles, dummy_weights)["params"]

        # Initialize decoder network
        dummy_encoding = encoder.apply({"params": encoder_params}, dummy_particles, dummy_weights)
        decoder_params = decoder.init(decoder_key, dummy_encoding)["params"]

        # Merge parameters
        params = {"encoder": encoder_params, "decoder": decoder_params}
        return params

    return AttentionPolicy(
        dim=decoder.dim,
        init=init,
        reset=reset,
        sample=sample,
        log_prob=log_prob,
        sample_and_log_prob=sample_and_log_prob,
        entropy=entropy,
    )


@partial(jax.jit, static_argnames="policy")
def train_attention_policy(
    learner: TrainState,
    policy: AttentionPolicy,
    actions: Array,
    particles: Array,
    weights: Array,
) -> tuple[TrainState, Array]:
    """Trains an attention policy using gradient-based optimization.

    Args:
        learner (TrainState): The current state of the training process
        policy (Policy): The policy to be trained
        actions (Array): The array of actions used for computing the log probabilities
        particles (Array): The array of particle sets
        weights (Array): The array of particle weights

    Returns:
        tuple[TrainState, Array]: A tuple containing the updated learner's
        training state and the computed loss value
    """
    def loss_fn(params):
        log_probs = policy.log_prob(actions, particles, weights, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss
