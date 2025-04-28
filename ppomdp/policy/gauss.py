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
    RecurrentPolicy,
    RecurrentObservation,
)
from ppomdp.policy.arch import (
    RecurrentEncoder,
    NeuralGaussDecoder
)


def create_recurrent_neural_gauss_policy(
    encoder: RecurrentEncoder,
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
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        next_actions = dist.sample(seed=rng_key)
        return next_carry, next_actions, bijector.forward(mean)

    def log_prob(
        next_actions: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> Array:
        _, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return dist.log_prob(next_actions)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array, Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        next_actions, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return next_carry, next_actions, log_prob, bijector.forward(mean)

    def carry_and_log_prob(
        next_actions: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        log_probs = dist.log_prob(next_actions)
        return next_carry, log_probs

    @jax.jit
    def pathwise_carry(
        actions: Array,
        observations: Array,
        params: Parameters,
    ):
        def body(carry, args):
            obs, act = args
            next_carry, _ = encoder.apply({"params": params["encoder"]}, carry, obs, act)
            return next_carry, next_carry

        _, batch_size, _ = observations.shape
        init_carry = encoder.reset(batch_size)
        _, all_carry = jax.lax.scan(body, init_carry, (observations, actions))

        def concat_trees(x, y):
            return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

        return concat_trees(init_carry, all_carry)

    @jax.jit
    def pathwise_log_prob(
        actions: Array,
        observations: Array,
        params: Parameters
    ) -> Array:

        def log_prob_fn(carry, args):
            _next_actions, _observations, _actions = args
            next_carry, log_prob_incs = \
                carry_and_log_prob(
                    next_actions=_next_actions,
                    carry=carry,
                    actions=_actions,
                    observations=_observations,
                    params=params
                )
            return next_carry, log_prob_incs

        _, batch_size, _ = actions.shape
        init_carry = encoder.reset(batch_size)

        _, log_probs = jax.lax.scan(
            f=log_prob_fn,
            init=init_carry,
            xs=(
                actions[1:, ...],
                observations[:-1, ...],
                actions[:-1, ...]
            )
        )
        return jnp.sum(log_probs, axis=0)

    def entropy(params: Parameters) -> Array:
        log_std = params["decoder"]["log_std"]
        return decoder.entropy(log_std)

    def reset(batch_size: int,) -> list[Carry]:
        return encoder.reset(batch_size)

    def init(
        rng_key: PRNGKey,
        obs_dim: int,
        action_dim: int,
        batch_dim: int,
    ) -> Parameters:
        dummy_key, encoder_key, decoder_key, _ = random.split(rng_key, 4)

        # initialize encoder network
        dummy_carry = encoder.reset(batch_dim)
        dummy_action = random.normal(dummy_key, (batch_dim, action_dim))
        dummy_observation = random.normal(dummy_key, (batch_dim, obs_dim))
        encoder_params = encoder.init(encoder_key, dummy_carry, dummy_observation, dummy_action)["params"]

        # initialize decoder network
        _, dummy_encoding = encoder.apply({"params": encoder_params}, dummy_carry, dummy_observation, dummy_action)
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
    learner: TrainState,
    policy: RecurrentPolicy,
    actions: Array,
    observations: Array,
) -> tuple[TrainState, Array]:
    """
    Trains a recurrent neural Gaussian policy using a pathwise gradient-based
    approach. This function optimizes the policy parameters by maximizing the
    log probability of the observed actions given the observations.

    Args:
        learner (TrainState): The current state of the training process,
            including model parameters and optimizers.
        policy (RecurrentPolicy): The policy to be trained, which must define
            the necessary methods for computing pathwise log probabilities.
        actions (Array): The array of actions used for computing the log probabilities.
        observations (Array): The array of observations corresponding to the actions.

    Returns:
        tuple[TrainState, Array]: A tuple containing the updated learner's
        training state and the computed loss value.
    """
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(actions, observations, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss


def initialize_multihead_recurrent_gauss_policy(
    rng_key: jax.random.PRNGKey,
    obs_dim: int,
    action_dim: int,
    batch_dim: int,
    encoder: RecurrentEncoder,
    prior_decoder: NeuralGaussDecoder,
    posterior_decoder: NeuralGaussDecoder,
) -> dict:
    """Initialize a multihead policy with shared encoder and separate decoders.

    Args:
        rng_key: Random key for initialization
        obs_dim: Observation dimension
        action_dim: Action dimension
        batch_dim: Batch dimension
        encoder: Shared encoder network
        prior_decoder: Decoder for prior policy
        posterior_decoder: Decoder for posterior policy

    Returns:
        tuple[RecurrentPolicy, RecurrentPolicy, dict]:
            - Prior policy
            - Posterior policy
            - Combined parameters dictionary with shared encoder
    """
    # Split keys for all operations
    dummy_key, encoder_key, prior_key, posterior_key = random.split(rng_key, 4)

    # Initialize encoder once
    dummy_carry = encoder.reset(batch_dim)
    dummy_action = random.normal(dummy_key, (batch_dim, action_dim))
    dummy_observation = random.normal(dummy_key, (batch_dim, obs_dim))
    encoder_params = encoder.init(encoder_key, dummy_carry, dummy_observation, dummy_action)["params"]

    # Get encoder output for decoder initialization
    _, dummy_encoding = encoder.apply({"params": encoder_params}, dummy_carry, dummy_observation, dummy_action)

    # Initialize decoders
    prior_decoder_params = prior_decoder.init(prior_key, dummy_encoding)["params"]
    posterior_decoder_params = posterior_decoder.init(posterior_key, dummy_encoding)["params"]

    # Combine parameters with shared encoder
    joint_params = {
        "encoder": encoder_params,
        "prior_decoder": prior_decoder_params,
        "posterior_decoder": posterior_decoder_params
    }
    return joint_params


@partial(jax.jit, static_argnames=("policy_prior", "policy_posterior"))
def train_multihead_recurrent_neural_gauss_policy_pathwise(
    learner: TrainState,
    policy_prior: RecurrentPolicy,
    policy_posterior: RecurrentPolicy,
    actions: Array,
    observations: Array,
    damping: float = 1.0
) -> tuple[TrainState, Array]:
    """Train a multihead recurrent neural Gaussian policy using pathwise gradients.

    This function performs one training step for a multihead recurrent neural network
    policy with Gaussian distribution outputs. It combines prior and posterior policies
    using weighted log probabilities according to the damping parameter.

    The training objective optimizes a weighted combination:
    log p(a) = (1-λ) log p_prior(a) + λ log p_posterior(a)
    where λ is the damping parameter.

    When damping=0, this reduces to training with only the prior policy.
    When damping=1, this reduces to training with only the posterior policy.

    Args:
        learner: Current training state with optimizer and parameters
        policy_prior: Prior policy model
        policy_posterior: Posterior policy model
        actions (Array): The array of actions used for computing the log probabilities.
        observations (Array): The array of observations corresponding to the actions.
        damping: Interpolation factor λ between prior and posterior policies.
                Must be in [0, 1]. Defaults to 1.0.

    Returns:
        tuple[TrainState, Array]:
            - Updated training state with new parameters
            - Scalar loss value

    Note:
        The parameters in the learner should have the structure:
        {
            "encoder": shared_encoder_params,
            "prior_decoder": prior_decoder_params,
            "posterior_decoder": posterior_decoder_params
        }
    """
    def loss_fn(params):
        prior_param = {"encoder": params["encoder"], "decoder": params["prior_decoder"]}
        posterior_param = {"encoder": params["encoder"], "decoder": params["posterior_decoder"]}

        prior_log_probs = policy_prior.pathwise_log_prob(actions, observations, prior_param)
        posterior_log_probs = policy_posterior.pathwise_log_prob(actions, observations, posterior_param)
        log_probs = (1. - damping) * prior_log_probs + damping * posterior_log_probs
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss



def create_recurrent_neural_gauss_observation(
    encoder: RecurrentEncoder,
    decoder: NeuralGaussDecoder,
) -> RecurrentObservation:

    def sample(
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings, next_actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sample = dist.sample(seed=rng_key)
        return next_carry, sample

    def log_prob(
        next_observations: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> Array:
        _, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings, next_actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return dist.log_prob(next_observations)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings, next_actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        sample, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return next_carry, sample, log_prob

    def carry_and_log_prob(
        next_observations: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        next_actions: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, encodings = encoder.apply({"params": params["encoder"]}, carry, observations, actions)
        mean, log_std = decoder.apply({"params": params["decoder"]}, encodings, next_actions)
        dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return next_carry, dist.log_prob(next_observations)

    def reset(batch_size: int,) -> list[Carry]:
        return encoder.reset(batch_size)

    def init(
        rng_key: PRNGKey,
        obs_dim: int,
        action_dim: int,
        batch_dim: int,
    ) -> Parameters:
        obs_key, action_key, encoder_key, decoder_key = random.split(rng_key, 4)

        # initialize encoder network
        dummy_carry = encoder.reset(batch_dim)
        dummy_observation = random.normal(obs_key, (batch_dim, obs_dim))
        dummy_action = random.normal(action_key, (batch_dim, action_dim))
        encoder_params = encoder.init(encoder_key, dummy_carry, dummy_observation, dummy_action)["params"]

        # initialize decoder network
        _, dummy_encoding = encoder.apply({"params": encoder_params}, dummy_carry, dummy_observation, dummy_action)
        decoder_params = decoder.init(decoder_key, dummy_encoding, dummy_action)["params"]

        # merge parameters
        params = {"encoder": encoder_params, "decoder": decoder_params}
        return params

    return RecurrentObservation(
        dim=decoder.dim,
        init=init,
        reset=reset,
        sample=sample,
        log_prob=log_prob,
        sample_and_log_prob=sample_and_log_prob,
        carry_and_log_prob=carry_and_log_prob,
    )


# @partial(jax.jit, static_argnames="observation_posterior")
# def train_recurrent_neural_gauss_observation_stepwise(
#     learner: TrainState,
#     observation_posterior: RecurrentObservation,
#     next_observations: Array,
#     carry: List[Carry],
#     actions: Array,
#     observations: Array,
#     next_actions: Array,
#     damping: float = 1.0
# ):
#
#     def loss_fn(params):
#         posterior_log_prob_observations = observation_posterior.log_prob(
#             next_observations=next_observations,
#             carry=carry,
#             actions=actions,
#             observations=observations,
#             next_actions=next_actions,
#             params=params
#         )
#         log_probs = damping * posterior_log_prob_observations
#         return -1.0 * jnp.mean(log_probs)
#
#     loss, grads = jax.value_and_grad(loss_fn)(learner.params)
#     learner = learner.apply_gradients(grads=grads)
#     return learner, loss


# @partial(
#     jax.jit,
#     static_argnames=(
#         "policy_prior",
#         "policy_posterior",
#         "observation_posterior"
#     )
# )
# def train_multihead_recurrent_neural_gauss_all_stepwise(
#     learner: TrainState,
#     policy_prior: RecurrentPolicy,
#     policy_posterior: RecurrentPolicy,
#     observation_posterior: RecurrentObservation,
#     next_actions: Array,
#     next_observations: Array,
#     carry: List[Carry],
#     actions: Array,
#     observations: Array,
#     damping: float = 1.0
# ):
#
#     def loss_fn(params):
#         policy_prior_param = {
#             "encoder": params["encoder"],
#             "decoder": params["policy_prior_decoder"]
#         }
#         policy_posterior_param = {
#             "encoder": params["encoder"],
#             "decoder": params["policy_posterior_decoder"]
#         }
#         observation_posterior_param = {
#             "encoder": params["encoder"],
#             "decoder": params["observation_posterior_decoder"]
#         }
#
#         prior_log_prob_actions = policy_prior.log_prob(
#             next_actions=next_actions,
#             carry=carry,
#             actions=actions,
#             observations=observations,
#             params=policy_prior_param
#         )
#         posterior_log_prob_actions = policy_posterior.log_prob(
#             next_actions=next_actions,
#             carry=carry,
#             actions=actions,
#             observations=observations,
#             params=policy_posterior_param
#         )
#         posterior_log_prob_observations = observation_posterior.log_prob(
#             next_observations=next_observations,
#             carry=carry,
#             actions=actions,
#             observations=observations,
#             next_actions=next_actions,
#             params=observation_posterior_param
#         )
#         log_probs = (1. - damping) * prior_log_prob_actions \
#                     + damping * posterior_log_prob_actions \
#                     + damping * posterior_log_prob_observations
#         return -1.0 * jnp.mean(log_probs)
#
#     loss, grads = jax.value_and_grad(loss_fn)(learner.params)
#     learner = learner.apply_gradients(grads=grads)
#     return learner, loss


@partial(jax.jit, static_argnames="policy")
def train_recurrent_neural_gauss_policy_pathwise_weighted(
    learner: TrainState,
    policy: RecurrentPolicy,
    actions: Array,
    observations: Array,
    importance_weights: Array,
) -> tuple[TrainState, Array]:
    """
    Trains a recurrent neural Gaussian policy pathwise using importance sampling.

    This function computes the loss based on the pathwise policy log probabilities,
    weighted by importance sampling ratios (clipped for stability), and updates
    the learner's parameters using gradient descent.
    Assumes pathwise_log_prob handles carry internally.

    Args:
        learner (TrainState): The training state containing model parameters
            and optimizer state.
        policy (RecurrentPolicy): The recurrent policy being trained.
        actions (Array): Actions tensor with shape (T, batch_size, action_dim).
        observations (Array): Observations tensor with shape (T, batch_size, obs_dim).
        importance_weights (Array): Importance weights per trajectory

    Returns:
        Tuple[TrainState, Array]: Updated learner state and the computed loss.
    """
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(actions, observations, params)
        return -1.0 * jnp.average(log_probs, weights=importance_weights)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss
