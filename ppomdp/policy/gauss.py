from functools import partial
from typing import Union, List

import jax

from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import (
    Bijector,
    MultivariateNormalDiag,
    Transformed,
)

from ppomdp.core import (
    PRNGKey,
    Carry,
    Parameters,
    HistoryParticles,
    RecurrentPolicy,
)
from ppomdp.policy.arch import (
    LSTMEncoder,
    GRUEncoder,
    NeuralGaussDecoder
)


def create_recurrent_neural_gauss_policy(
    encoder: Union[LSTMEncoder, GRUEncoder],
    decoder: NeuralGaussDecoder,
    bijector: Bijector
) -> RecurrentPolicy:
    r"""Creates a squashed neural Gaussian policy that conforms to the RecurrentPolicy interface.

    The policy uses a recurrent encoder to process observations and a decoder to output
    action distributions. Actions are transformed through a bijector to enforce bounds.

    Args:
        encoder (Union[LSTMEncoder, GRUEncoder]): The recurrent encoder network
        decoder (NeuralGaussDecoder): The neural network used for the policy
        bijector (Bijector): Policy bijector to enforce action limits

    Returns:
        RecurrentPolicy: A policy object implementing the RecurrentPolicy interface with
            methods for sampling actions and computing probabilities
    """

    def sample(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        sample = dist.sample(seed=rng_key)
        return next_carry, sample, bijector.forward(mean)

    def log_prob(
        actions: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> Array:
        _, encodings = encoder.apply({"params": params["encoder"]}, carry, observations)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return dist.log_prob(actions)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array, Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        sample, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return next_carry, sample, log_prob, bijector.forward(mean)

    def carry_and_log_prob(
        action: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        log_prob = dist.log_prob(action)
        return next_carry, log_prob

    @jax.jit
    def pathwise_carry(
        init_carry: list[Carry],
        observations: Array,
        params: Parameters,
    ):
        def body(carry, obs):
            next_carry, _ = encoder.apply({"params": params["encoder"]}, carry, obs)
            return next_carry, next_carry

        _, all_carry = jax.lax.scan(body, init_carry, observations)

        def concat_trees(x, y):
            return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

        return concat_trees(init_carry, all_carry)

    def pathwise_log_prob(
        particles: HistoryParticles,
        params: Parameters
    ) -> Array:
        def body(t, log_probs):
            actions = particles.actions[t]
            observations = particles.observations[t - 1]
            carry = jax.tree.map(lambda x: x[t - 1], particles.carry)
            log_prob_incs = log_prob(actions, carry, observations, params)
            return log_probs + log_prob_incs

        num_time_steps, batch_size, _ = particles.actions.shape
        init_log_probs = jnp.zeros(batch_size)
        log_probs = jax.lax.fori_loop(1, num_time_steps, body, init_log_probs)
        return log_probs

    def entropy(params: Parameters) -> Array:
        log_std = params["decoder"]["log_std"]
        return decoder.entropy(log_std)

    def reset(batch_size: int,) -> list[Carry]:
        return encoder.reset(batch_size)

    def init(
        rng_key: PRNGKey,
        obs_dim: int,
        batch_dim: int,
    ) -> Parameters:
        obs_key, encoder_key, decoder_key = random.split(rng_key, 3)

        # initialize encoder network
        dummy_carry = encoder.reset(batch_dim)
        dummy_observation = random.normal(obs_key, (batch_dim, obs_dim))
        encoder_params = encoder.init(encoder_key, dummy_carry, dummy_observation)["params"]

        # initialize decoder network
        _, dummy_encoding = encoder.apply({"params": encoder_params}, dummy_carry, dummy_observation)
        decoder_params = decoder.init(decoder_key, dummy_encoding)["params"]

        # merge parameters
        params = {"encoder": encoder_params, "decoder": decoder_params}
        return params

    return RecurrentPolicy(
        dim=decoder.dim,
        init=init,
        reset=reset,
        sample=sample,
        log_prob=log_prob,
        pathwise_carry=pathwise_carry,
        pathwise_log_prob=pathwise_log_prob,
        sample_and_log_prob=sample_and_log_prob,
        carry_and_log_prob=carry_and_log_prob,
        entropy=entropy,
    )


@partial(jax.jit, static_argnames="policy")
def train_recurrent_neural_gauss_policy_pathwise(
    policy: RecurrentPolicy,
    learner: TrainState,
    particles: HistoryParticles,
    damping: float = 1.0
) -> tuple[TrainState, Array]:
    """
    Trains a recurrent neural Gaussian policy using a pathwise gradient-based
    approach. This function optimizes the policy parameters through gradient
    updates with respect to the provided particles and current state of the
    learner.

    Args:
        policy (RecurrentPolicy): The policy to be trained, which must define
            the necessary methods for computing pathwise log probabilities.
        learner (TrainState): The current state of the training process,
            including model parameters and optimizers.
        particles (HistoryParticles): The data structure representing
            the history particles over which the policy's log probability
            is computed.
        damping (float): A scaling factor applied to the computed log probabilities
            during the gradient computation. Defaults to 1.0.

    Returns:
        tuple[TrainState, Array]: A tuple containing the updated learner's
        training state and the computed loss value.
    """
    def loss_fn(params):
        log_probs = damping * policy.pathwise_log_prob(particles, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss


@partial(jax.jit, static_argnames="policy")
def train_recurrent_neural_gauss_policy_stepwise(
    policy: RecurrentPolicy,
    learner: TrainState,
    actions: Array,
    carry: List[Carry],
    observations: Array,
    damping: float = 1.0
):
    r"""Performs a single training step for the recurrent neural Gaussian policy.

    Args:
        policy (RecurrentPolicy): The policy to be trained
        learner (TrainState): Current training state
        actions (Array): Batch of actions
        carry (List[Carry]): List of recurrent state carries
        observations (Array): Batch of observations
        damping (float, optional): Damping factor for the loss. Defaults to 1.0.

    Returns:
        tuple[TrainState, float]: Updated training state and loss value
    """
    def loss_fn(params):
        log_probs = damping * policy.log_prob(
            actions=actions,
            carry=carry,
            observations=observations,
            params=params
        )
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss


@partial(jax.jit, static_argnames=("policy_prior", "policy_posterior"))
def train_multihead_recurrent_neural_gauss_policy_stepwise(
    policy_prior: RecurrentPolicy,
    policy_posterior: RecurrentPolicy,
    learner: TrainState,
    actions: Array,
    carry: List[Carry],
    observations: Array,
    damping: float = 1.0
):
    r"""Performs a single training step for the multi-head recurrent neural Gaussian policy.

    The function implements a weighted combination of prior and posterior policies using the formula:
    $\log p(a) = (1-\lambda) \log p_{\text{prior}}(a) + \lambda \log p_{\text{posterior}}(a)$
    where $\lambda$ is the damping factor.

    Args:
        policy_prior (RecurrentPolicy): The prior policy model
        policy_posterior (RecurrentPolicy): The posterior policy model
        learner (TrainState): Current training state containing optimizer state and parameters
        actions (Array): Batch of actions with shape (batch_size, action_dim)
        carry (List[Carry]): List of recurrent state carries containing hidden states
        observations (Array): Batch of observations with shape (batch_size, obs_dim)
        damping (float, optional): Interpolation factor $\lambda$ between prior and posterior.
            Defaults to 1.0.

    Returns:
        tuple[TrainState, float]: A tuple containing:
            - Updated training state with new parameters
            - Scalar loss value of the current training step
    """

    def loss_fn(params):
        prior_param = {"encoder": params["encoder"], "decoder": params["prior_decoder"]}
        posterior_param = {"encoder": params["encoder"], "decoder": params["posterior_decoder"]}

        prior_log_probs = policy_prior.log_prob(
            actions=actions,
            carry=carry,
            observations=observations,
            params=prior_param
        )
        posterior_log_probs = policy_posterior.log_prob(
            actions=actions,
            carry=carry,
            observations=observations,
            params=posterior_param
        )
        log_probs = (1. - damping) * prior_log_probs + damping * posterior_log_probs
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss