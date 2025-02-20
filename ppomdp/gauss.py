from functools import partial
from typing import Callable, Union

import jax
import optax

from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from distrax import (
    Chain,
    MultivariateNormalDiag,
    Transformed,
    Block
)

from ppomdp.core import (
    Carry,
    RecurrentPolicy,
    Parameters,
    OuterParticles,
    PRNGKey
)
from ppomdp.arch import (
    LSTMEncoder,
    GRUEncoder,
    MLPDecoder
)


class RecurrentNeuralGauss(nn.Module):
    """
    Neural policy module for processing sequences with recurrent encoding and dense decoding layers.

    Attributes:
        encoder (Union[ppomdp.arch.LSTMEncoder, ppomdp.arch.GRUEncoder]): Recurrent encoder module.
        decoder (Decoder): Dense decoder module.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
    """

    encoder: Union[LSTMEncoder, GRUEncoder]
    decoder: MLPDecoder
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, carry: list[Carry], s: Array) -> tuple[list[Carry], Array]:
        log_std = self.param("log_std", self.init_log_std, self.decoder.output_dim)

        carry, s = self.encoder(carry, s)
        s = self.decoder(s)
        return carry, s

    @property
    def dim(self):
        return self.decoder.output_dim

    def reset(self, batch_size: int) -> list[Carry]:
        return self.encoder.reset(batch_size)


def create_recurrent_gauss_policy(
    network: RecurrentNeuralGauss,
    bijector: Chain
) -> RecurrentPolicy:
    """
    Creates a squashed neural policy that conforms to the RecurrentPolicy interface.

    Args:
        network (RecurrentNeuralGauss): The neural network used for the policy
        bijector (Chain): policy bijector to enforce action limits

    Returns:
        RecurrentPolicy: A policy that implements the RecurrentPolicy interface
    """

    def reset(batch_size: int) -> list[Carry]:
        return network.reset(batch_size)

    def sample(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        carry, mean = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.dim))
        return carry, dist.sample(seed=rng_key)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        carry, mean = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.dim))
        action, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return carry, action, log_prob

    def carry_and_log_prob(
        action: Array,
        carry: list[Carry],
        observation: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        carry, mean = network.apply({"params": params}, carry, observation)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.dim))
        return carry, dist.log_prob(action)

    def log_prob(
        actions: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> Array:
        _, mean = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.dim))
        return dist.log_prob(actions)

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
        sigma = jnp.diag(jnp.exp(2. * params["log_std"]))
        return 0.5 * (
                network.dim * jnp.log(2.0 * jnp.pi * jnp.exp(1))
                + jnp.linalg.slogdet(sigma)[1]
        )

    def init(
        rng_key: PRNGKey,
        input_dim: int,
        output_dim: int,
        batch_dim: int,
        learning_rate: float,
    ) -> TrainState:
        input_key, param_key = random.split(rng_key, 2)
        dummy_carry = network.reset(batch_dim)
        dummy_input = random.normal(input_key, (batch_dim, input_dim))
        init_params = network.init(param_key, dummy_carry, dummy_input)["params"]
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=optax.adam(learning_rate)
        )
        return train_state

    return RecurrentPolicy(
        dim=network.dim,
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
def train_recurrent_gauss_policy(
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
