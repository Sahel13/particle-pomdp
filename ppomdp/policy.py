from functools import partial
from typing import Callable, Sequence, Union

import jax
from jax import Array, random
import jax.numpy as jnp
from chex import PRNGKey

from distrax import Block, Chain, MultivariateNormalDiag, Transformed
from flax import linen as nn
from flax.training.train_state import TrainState

import optax

from ppomdp.core import (
    LSTMCarry,
    GRUCarry,
    Carry,
    Parameters,
    OuterParticles,
    RecurrentPolicy
)


class LSTMEncoding(nn.Module):
    """
    LSTM module for processing sequences with optional feature extraction and encoding layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
    """

    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]

    @nn.compact
    def __call__(self, carry: list[LSTMCarry], s: Array) -> tuple[list[LSTMCarry], Array]:
        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for size in self.encoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, size in enumerate(self.recurr_size):
            carry[k], y = nn.LSTMCell(size)(carry[k], y)

        return carry, y

    def reset(self, batch_size) -> list[LSTMCarry]:
        carry = []
        for size in self.recurr_size:
            mem_shape = (batch_size, size)
            c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
            carry.append((c, h))
        return carry


class GRUEncoding(nn.Module):
    """
    GRU module for processing sequences with optional feature extraction and encoding layers.

    Attributes:
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
    """

    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]

    @nn.compact
    def __call__(self, carry: list[GRUCarry], s: Array) -> tuple[list[GRUCarry], Array]:
        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for size in self.encoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, size in enumerate(self.recurr_size):
            carry[k], y = nn.GRUCell(size)(carry[k], y)

        return carry, y

    def reset(self, batch_size) -> list[GRUCarry]:
        carry = []
        for size in self.recurr_size:
            mem_shape = (batch_size, size)
            h = jnp.zeros(mem_shape)  # GRUCarry
            carry.append(h)
        return carry


class NeuralPolicy(nn.Module):

    encoder: Union[LSTMEncoding, GRUEncoding]
    decoder_size: Sequence[int]
    output_size: int
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, carry: list[Carry], s: Array) -> tuple[list[Carry], Array]:
        log_std = self.param("log_std", self.init_log_std, self.output_size)

        # pass inputs through features layer
        carry, y = self.encoder(carry, s)

        # pass result through decoder layers
        for size in self.decoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.output_size)(y)
        return carry, y

    def reset(self, batch_size: int) -> list[Carry]:
        return self.encoder.reset(batch_size)


def create_neural_policy(network: NeuralPolicy, bijector: Chain) -> RecurrentPolicy:
    """
    Creates a squashed neural policy that conforms to the RecurrentPolicy interface.

    Args:
        network (NeuralPolicy): The neural network used for the policy
        bijector (Chain): The bijector used to transform the distribution

    Returns:
        RecurrentPolicy: A policy that implements the RecurrentPolicy interface
    """

    def reset(batch_size: int) -> list[Carry]:
        return network.reset(batch_size)

    def sample(
        rng_key: PRNGKey,
        observations: Array,
        carry: list[Carry],
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        carry, mean = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.output_size))
        return carry, dist.sample(seed=rng_key)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        observations: Array,
        carry: list[Carry],
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        carry, mean = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.output_size))
        action, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return carry, action, log_prob

    def carry_and_log_prob(
        action: Array,
        observation: Array,
        carry: list[Carry],
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, mean = network.apply({"params": params}, carry, observation)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.output_size))
        return next_carry, dist.log_prob(action)

    def log_prob(
        actions: Array,
        observations: Array,
        carry: list[Carry],
        params: Parameters,
    ) -> Array:
        _, mean = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(params["log_std"]))
        dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.output_size))
        return dist.log_prob(actions)

    def pathwise_log_prob(
        particles: OuterParticles,
        params: Parameters
    ) -> Array:
        def body(t, log_probs):
            actions = particles.actions[t]
            observations = particles.observations[t - 1]
            carry = jax.tree.map(lambda x: x[t - 1], particles.carry)
            log_prob_incs = log_prob(actions, observations, carry, params)
            return log_probs + log_prob_incs

        num_time_steps, batch_size, _ = particles.actions.shape
        init_log_probs = jnp.zeros(batch_size)
        log_probs = jax.lax.fori_loop(1, num_time_steps, body, init_log_probs)
        return log_probs

    def entropy(params: Parameters) -> Array:
        sigma = jnp.diag(jnp.exp(2. * params["log_std"]))
        return 0.5 * (
                network.output_size * jnp.log(2.0 * jnp.pi * jnp.exp(1))
                + jnp.linalg.slogdet(sigma)[1]
        )

    def init(
        rng_key: PRNGKey,
        input_size: int,
        batch_size: int,
        learning_rate: float,
    ) -> TrainState:
        input_key, param_key = random.split(rng_key, 2)
        init_carry = network.reset(batch_size)
        init_input = random.normal(input_key, (batch_size, input_size))
        init_params = network.init(param_key, init_carry, init_input)["params"]
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=optax.adam(learning_rate)
        )
        return train_state

    return RecurrentPolicy(
        dim=network.output_size,
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
def train_neural_policy(
    policy: RecurrentPolicy,
    train_state: TrainState,
    particles: OuterParticles,
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        log_probs = policy.pathwise_log_prob(particles, params)
        return -1.0 * jnp.mean(log_probs)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


# def reset_policy(batch_size: int, network: Union[LSTM, GRU]) -> list[Carry]:
#     return network.reset(batch_size)
#
#
# def squash_policy(mean: Array, log_std: Array, bijector: Chain) -> Transformed:
#     dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
#     return Transformed(distribution=dist, bijector=Block(bijector, ndims=1))
#
#
# def sample_policy(
#     rng_key: PRNGKey,
#     observations: Array,
#     carry: list[Carry],
#     params: Parameters,
#     network: Union[LSTM, GRU],
#     bijector: Chain,
# ) -> tuple[list[Carry], Array]:
#     carry, m = network.apply({"params": params}, carry, observations)
#     dist = squash_policy(m, params["log_std"], bijector)
#     return carry, dist.sample(seed=rng_key)
#
#
# def sample_and_log_prob_policy(
#     rng_key: PRNGKey,
#     observations: Array,
#     carry: list[Carry],
#     params: Parameters,
#     network: Union[LSTM, GRU],
#     bijector: Chain,
# ) -> tuple[list[Carry], Array, Array]:
#     carry, m = network.apply({"params": params}, carry, observations)
#     dist = squash_policy(m, params["log_std"], bijector)
#     action, log_prob = dist.sample_and_log_prob(seed=rng_key)
#     return carry, action, log_prob
#
#
# def carry_and_log_prob_policy(
#     action: Array,
#     observation: Array,
#     carry: list[Carry],
#     params: Parameters,
#     network: Union[LSTM, GRU],
#     bijector: Chain,
# ) -> tuple[list[Carry], Array]:
#     next_carry, m = network.apply({"params": params}, carry, observation)
#     dist = squash_policy(m, params["log_std"], bijector)
#     return next_carry, dist.log_prob(action)
#
#
# def log_prob_policy(
#     actions: Array,
#     observations: Array,
#     carry: list[Carry],
#     params: Parameters,
#     network: Union[LSTM, GRU],
#     bijector: Chain,
# ) -> Array:
#     _, m = network.apply({"params": params}, carry, observations)
#     dist = squash_policy(m, params["log_std"], bijector)
#     return dist.log_prob(actions)
#
#
# def entropy_policy(
#     params: Parameters,
#     network: Union[LSTM, GRU],
#     bijector: Chain,
# ) -> Array:
#     sigma = jnp.diag(jnp.exp(2. * params["log_std"]))
#     return 0.5 * (
#         network.output_dim * jnp.log(2.0 * jnp.pi * jnp.exp(1))
#         + jnp.linalg.slogdet(sigma)[1]
#     )
#
#
# def get_recurrent_policy(network: Union[LSTM, GRU], bijector: Chain):
#     return RecurrentPolicy(
#         dim=network.output_dim,
#         reset=partial(reset_policy, network=network),
#         sample=partial(sample_policy, network=network, bijector=bijector),
#         log_prob=partial(log_prob_policy, network=network, bijector=bijector),
#         sample_and_log_prob=partial(sample_and_log_prob_policy, network=network, bijector=bijector),
#         carry_and_log_prob=partial(carry_and_log_prob_policy, network=network, bijector=bijector),
#         entropy=partial(entropy_policy, network=network, bijector=bijector),
#     )
