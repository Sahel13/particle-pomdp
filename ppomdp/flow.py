from functools import partial
from typing import Union, Callable

import jax
import optax

from jax import Array, random, numpy as jnp

from flax import linen as nn
from flax.training.train_state import TrainState

from distrax import Bijector, MultivariateNormalDiag

from ppomdp.core import (
    Carry,
    Parameters,
    RecurrentPolicy,
    OuterParticles,
    PRNGKey
)
from ppomdp.arch import (
    LSTMEncoder,
    GRUEncoder,
    MLPConditioner,
    MLPDecoder
)
from ppomdp.bijector import (
    ChainConditional,
    InverseConditional,
    MaskedCouplingConditional,
    TransformedConditional
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
    encoder: Union[LSTMEncoder, GRUEncoder]
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

    def __call__(self, x: Array, carry: list[Carry], s: Array) -> Array:
        _, s = self.encoder(carry, s)
        s = self.decoder(s)
        return self.dist.log_prob(x, context=s)

    def sample(self, rng_key: PRNGKey, carry: list[Carry], s: Array) -> tuple[list[Carry], Array]:
        carry, s = self.encoder(carry, s)
        s = self.decoder(s)
        x = self.dist.sample(seed=rng_key, context=s, sample_shape=s.shape[0])
        return carry, x

    def sample_and_log_prob(self, rng_key: PRNGKey, carry: list[Carry], s: Array) -> tuple[list[Carry], Array, Array]:
        carry, s = self.encoder(carry, s)
        s = self.decoder(s)
        x, log_prob = self.dist.sample_and_log_prob(seed=rng_key, context=s, sample_shape=s.shape[0])
        return carry, x, log_prob

    def carry_and_log_prob(self, x: Array, carry: list[Carry], s: Array) -> tuple[list[Carry], Array]:
        carry, s = self.encoder(carry, s)
        s = self.decoder(s)
        return carry, self.dist.log_prob(x, context=s)

    def log_prob(self, x: Array, carry: list[Carry], s: Array) -> Array:
        _, s = self.encoder(carry, s)
        s = self.decoder(s)
        return self.dist.log_prob(x, context=s)

    def reset(self, batch_dim: int) -> list[Carry]:
        return self.encoder.reset(batch_dim)

    def entropy(self) -> Array:
        return self.base.entropy()


def create_recurrent_flow_policy(flow: RecurrentNeuralFlow) -> RecurrentPolicy:
    """
    Creates a policy from a recurrent neural flow that conforms to the RecurrentPolicy interface.

    Args:
        flow (RecurrentNeuralFlow): The recurrent neural flow used for the policy

    Returns:
        RecurrentPolicy: A policy that implements the RecurrentPolicy interface
    """

    def reset(batch_size: int) -> list[Carry]:
        return flow.reset(batch_size)

    def sample(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        carry, actions = flow.apply(
            {"params": params},
            method=flow.sample,
            rng_key=rng_key,
            carry=carry,
            s=observations,
        )
        return carry, actions

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        carry, actions, log_probs = flow.apply(
            {"params": params},
            method=flow.sample_and_log_prob,
            rng_key=rng_key,
            carry=carry,
            s=observations,
        )
        return carry, actions, log_probs

    def carry_and_log_prob(
        action: Array,
        carry: list[Carry],
        observation: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        carry, log_prob = flow.apply(
            {"params": params},
            method=flow.carry_and_log_prob,
            x=action,
            carry=carry,
            s=observation
        )
        return carry, log_prob

    def log_prob(
        actions: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> Array:
        return flow.apply(
            {"params": params},
            x=actions,
            carry=carry,
            s=observations
        )

    def pathwise_log_prob(
        particles: OuterParticles,
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
        return flow.apply(
            {"params": params},
            method=flow.entropy,
        )

    def init(
        rng_key: PRNGKey,
        input_dim: int,
        output_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        input_key, output_key, param_key = random.split(rng_key, 3)
        dummy_carry = flow.reset(batch_dim)
        dummy_input = random.normal(input_key, (batch_dim, input_dim))
        dummy_output = random.normal(output_key, (batch_dim, output_dim))
        init_params = flow.init(param_key, dummy_output, dummy_carry, dummy_input)["params"]
        train_state = TrainState.create(
            apply_fn=flow.apply,
            params=init_params,
            tx=optax.adam(learning_rate)
        )
        return train_state

    return RecurrentPolicy(
        dim=flow.dim,
        reset=reset,
        sample=sample,
        log_prob=log_prob,
        pathwise_log_prob=pathwise_log_prob,
        sample_and_log_prob=sample_and_log_prob,
        carry_and_log_prob=carry_and_log_prob,
        entropy=entropy,
        init=init
    )


@partial(jax.jit, static_argnums=(0,))
def train_recurrent_flow_policy(
    policy: RecurrentPolicy,
    train_state: TrainState,
    particles: OuterParticles,
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(particles, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss
