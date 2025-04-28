from functools import partial
from typing import Union, Callable

import jax
from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from distrax import MultivariateNormalDiag

from ppomdp.core import (
    PRNGKey,
    Carry,
    Parameters,
    RecurrentPolicy,
)
from ppomdp.bijector import (
    ChainConditional,
    InverseConditional,
    MaskedCouplingConditional,
    TransformedConditional
)
from ppomdp.policy.arch import (
    RecurrentEncoder,
    MLPConditioner,
    MLPDecoder
)


class RecurrentNeuralFlow(nn.Module):
    """
    Recurrent neural flow module for processing sequences with recurrent encoding and dense decoding layers.

    Attributes:
        dim (int): Dimensionality of the output.
        encoder (Union[LSTMEncoder, GRUEncoder]): Recurrent encoder module.
        decoder (MLPDecoder): Dense decoder module.
        conditioners (list[MLPConditioner]): List of conditioner modules for the masked coupling layers.
        inner_bijector (Callable): Inner bijector function for the masked coupling layers.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
    """

    dim: int
    encoder: RecurrentEncoder
    decoder: MLPDecoder
    conditioners: list[MLPConditioner]
    inner_bijector: Callable
    init_log_std: Callable = nn.initializers.ones

    def setup(self):

        mask = jnp.arange(0, self.dim) % 2
        mask = jnp.reshape(mask, self.dim)
        mask = mask.astype(bool)

        bijector_list = []
        for conditioner in self.conditioners:
            mask = jnp.logical_not(mask) if self.dim > 1 else jnp.array([False], dtype=bool)
            bijector_list.append(
                MaskedCouplingConditional(
                    mask=mask,
                    bijector=self.inner_bijector,
                    conditioner=conditioner
                )
            )

        from distrax import Transformed, Block
        from ppomdp.bijector import Tanh

        log_std = self.param("log_std", self.init_log_std, self.dim)
        _base = MultivariateNormalDiag(jnp.zeros(self.dim), jnp.exp(log_std))
        self.base = Transformed(_base, Block(Tanh(), ndims=1))

        self.bijector = InverseConditional(ChainConditional(bijector_list))
        self.dist = TransformedConditional(self.base, self.bijector)

    def __call__(self, x: Array, carry: list[Carry], z: Array, a: Array) -> Array:
        _, encodings = self.encoder(carry, z, a)
        y = self.decoder(encodings)
        return self.dist.log_prob(x, context=y)

    def sample(self, rng_key: PRNGKey, carry: list[Carry], z: Array, a: Array) -> tuple[list[Carry], Array]:
        next_carry, encodings = self.encoder(carry, z, a)
        y = self.decoder(encodings)
        x = self.dist.sample(seed=rng_key, context=y, sample_shape=z.shape[0])
        return next_carry, x

    def log_prob(self, x: Array, carry: list[Carry], z: Array, a: Array) -> Array:
        _, encodings = self.encoder(carry, z, a)
        y = self.decoder(encodings)
        return self.dist.log_prob(x, context=y)

    def sample_and_log_prob(self, rng_key: PRNGKey, carry: list[Carry], z: Array, a: Array) -> tuple[list[Carry], Array, Array]:
        next_carry, encodings = self.encoder(carry, z, a)
        y = self.decoder(encodings)
        x, log_prob = self.dist.sample_and_log_prob(seed=rng_key, context=y, sample_shape=z.shape[0])
        return next_carry, x, log_prob

    def carry_and_log_prob(self, x: Array, carry: list[Carry], z: Array, a: Array) -> tuple[list[Carry], Array]:
        next_carry, encodings = self.encoder(carry, z, a)
        y = self.decoder(encodings)
        return next_carry, self.dist.log_prob(x, context=y)

    def reset(self, batch_dim: int) -> list[Carry]:
        return self.encoder.reset(batch_dim)

    def entropy(self) -> Array:
        return self.base.entropy()


def create_recurrent_neural_flow_policy(
    flow: RecurrentNeuralFlow
) -> RecurrentPolicy:
    """
    Creates a wrapped recurrent neural flow policy with several auxiliary methods for sampling,
    log probability calculation, initialization, and entropy calculation. The returned policy object
    encapsulates various behaviors required for recurrent flow models and provides a high-level
    interface to interact with the underlying flow. This function binds specific input-output
    relations and logic to the given `flow` object.

    Args:
        flow: An instance of `RecurrentNeuralFlow` that defines the underlying logic for
            sampling, log probability computation, initialization, and other associated
            operations required by the policy.

    Returns:
        An instance of `RecurrentPolicy`, encapsulating methods for sampling, calculating log
        probabilities, initializing parameters, resetting the state, and entropy computation.
    """

    def sample(
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        next_carry, next_actions = flow.apply(
            {"params": params},
            method=flow.sample,
            rng_key=rng_key,
            carry=carry,
            z=observations,
            a=actions
        )
        return next_carry, next_actions, next_actions

    def log_prob(
        next_actions: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> Array:
        return flow.apply(
            {"params": params},
            x=next_actions,
            carry=carry,
            z=observations,
            a=actions
        )

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array, Array]:
        next_carry, next_actions, log_probs = flow.apply(
            {"params": params},
            method=flow.sample_and_log_prob,
            rng_key=rng_key,
            carry=carry,
            z=observations,
            a=actions
        )
        return next_carry, next_actions, next_actions, log_probs

    def carry_and_log_prob(
        next_actions: Array,
        carry: list[Carry],
        actions: Array,
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, log_prob = flow.apply(
            {"params": params},
            method=flow.carry_and_log_prob,
            x=next_actions,
            carry=carry,
            z=observations,
            a=actions
        )
        return next_carry, log_prob

    def pathwise_carry(
        actions: Array,
        observations: Array,
        params: Parameters,
    ):
        raise NotImplementedError

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
        init_carry = flow.reset(batch_size)

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

    def reset(batch_size: int) -> list[Carry]:
        return flow.reset(batch_size)

    def entropy(params: Parameters) -> Array:
        return flow.apply({"params": params}, method=flow.entropy)

    def init(
        rng_key: PRNGKey,
        obs_dim: int,
        action_dim: int,
        batch_dim: int,
    ) -> Parameters:
        obs_key, action_key, param_key = random.split(rng_key, 3)
        dummy_carry = flow.reset(batch_dim)
        dummy_observation = random.normal(obs_key, (batch_dim, obs_dim))
        dummy_action = random.normal(action_key, (batch_dim, action_dim))
        params = flow.init(param_key, dummy_action, dummy_carry, dummy_observation, dummy_action)["params"]
        return params

    return RecurrentPolicy(
        dim=flow.dim,
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
def train_recurrent_neural_flow_policy_pathwise(
    policy: RecurrentPolicy,
    learner: TrainState,
    actions: Array,
    observations: Array,
) -> tuple[TrainState, Array]:
    """
    Trains a recurrent neural flow policy using pathwise gradients with uniform weighting.

    This function computes the log probability of action sequences given observation
    sequences using the policy's pathwise log probability function, and then updates
    the policy parameters to maximize these log probabilities.

    Args:
        policy (RecurrentPolicy): The recurrent policy to be trained.
        learner (TrainState): The training state containing the current parameters and optimizer.
        actions (Array): Sequence of actions with shape [time_steps, batch_size, action_dim].
        observations (Array): Sequence of observations with shape [time_steps, batch_size, obs_dim].

    Returns:
        tuple[TrainState, Array]: A tuple containing:
            - The updated learner state with new parameters after the gradient update
            - The loss value (negative mean log probability) before the update
    """
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(actions, observations, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss


@partial(jax.jit, static_argnames="policy")
def train_recurrent_neural_flow_policy_pathwise_weighted(
    policy: RecurrentPolicy,
    learner: TrainState,
    actions: Array,
    observations: Array,
    importance_weights: Array,
) -> tuple[TrainState, Array]:
    """
    Trains a recurrent neural flow policy using pathwise gradients with importance weighting.

    Similar to the unweighted version, but incorporates importance weights to prioritize
    certain sequences during training. This is useful for off-policy learning or when
    some trajectories are more valuable for learning than others.

    Args:
        policy (RecurrentPolicy): The recurrent policy to be trained.
        learner (TrainState): The training state containing the current parameters and optimizer.
        actions (Array): Sequence of actions with shape [time_steps, batch_size, action_dim].
        observations (Array): Sequence of observations with shape [time_steps, batch_size, obs_dim].
        importance_weights (Array): Weights for each sequence in the batch with shape [batch_size].
            Higher weights give more importance to those sequences during training.

    Returns:
        tuple[TrainState, Array]: A tuple containing:
            - The updated learner state with new parameters after the gradient update
            - The loss value (negative weighted average log probability) before the update
    """
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(actions, observations, params)
        return -1.0 * jnp.average(log_probs, weights=importance_weights)

    loss, grads = jax.value_and_grad(loss_fn)(learner.params)
    learner = learner.apply_gradients(grads=grads)
    return learner, loss