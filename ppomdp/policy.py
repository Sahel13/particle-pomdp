from functools import partial
from typing import Callable, Dict, Sequence, Union

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Block, Chain, MultivariateNormalDiag, Transformed
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import Array

from ppomdp.core import (
    LSTMCarry,
    GRUCarry,
    Carry,
    OuterParticles,
    RecurrentPolicy
)


class LSTM(nn.Module):
    """
    LSTM module for processing sequences with optional feature extraction and encoding layers.

    Attributes:
        dim (int): Dimensionality of the output.
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
        output_size (Sequence[int]): Sizes of the output layers.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
    """

    dim: int
    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]
    output_size: Sequence[int]
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, carry: list[LSTMCarry], s: Array) -> tuple[list[LSTMCarry], Array]:
        log_std = self.param("log_std", self.init_log_std, self.dim)

        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for size in self.encoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, size in enumerate(self.recurr_size):
            carry[k], y = nn.LSTMCell(size)(carry[k], y)

        # pass result through output layers
        for size in self.output_size:
            y = nn.relu(nn.Dense(size)(y))
        a = nn.Dense(self.dim)(y)
        return carry, a


def reset_lstm(batch_size: int, network: LSTM) -> list[LSTMCarry]:
    carry = []
    for _size in network.recurr_size:
        mem_shape = (batch_size, _size)
        c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
        carry.append((c, h))
    return carry


class GRU(nn.Module):
    """
    GRU module for processing sequences with optional feature extraction and encoding layers.

    Attributes:
        dim (int): Dimensionality of the output.
        feature_fn (Callable): Function to extract features from the input sequence.
        encoder_size (Sequence[int]): Sizes of the encoding layers.
        recurr_size (Sequence[int]): Sizes of the recurrent layers.
        output_size (Sequence[int]): Sizes of the output layers.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
    """

    dim: int
    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]
    output_size: Sequence[int]
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, carry: list[GRUCarry], s: Array) -> tuple[list[GRUCarry], Array]:
        log_std = self.param("log_std", self.init_log_std, self.dim)

        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for size in self.encoder_size:
            y = nn.relu(nn.Dense(size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, size in enumerate(self.recurr_size):
            carry[k], y = nn.GRUCell(size)(carry[k], y)

        # pass result through output layers
        for size in self.output_size:
            y = nn.relu(nn.Dense(size)(y))
        a = nn.Dense(self.dim)(y)
        return carry, a


def reset_gru(batch_size: int, network: GRU) -> list[GRUCarry]:
    carry = []
    for _size in network.recurr_size:
        mem_shape = (batch_size, _size)
        h = jnp.zeros(mem_shape)  # GRUCarry
        carry.append(h)
    return carry


def reset_policy(batch_size: int, network: Union[LSTM, GRU]) -> list[Carry]:
    if isinstance(network, LSTM):
        return reset_lstm(batch_size, network)
    else:
        return reset_gru(batch_size, network)
    # return jax.lax.cond(
    #     isinstance(network, LSTM),
    #     lambda _: reset_lstm(batch_size, network),
    #     lambda _: reset_gru(batch_size, network),
    #     operand=None
    # )


def squash_policy(mean: Array, log_std: Array, bijector: Chain) -> Transformed:
    dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    return Transformed(distribution=dist, bijector=Block(bijector, ndims=1))


def sample_policy(
    rng_key: PRNGKey,
    observations: Array,
    carry: list[Carry],
    params: Dict,
    network: Union[LSTM, GRU],
    bijector: Chain,
) -> tuple[list[Carry], Array]:
    carry, m = network.apply({"params": params}, carry, observations)
    dist = squash_policy(m, params["log_std"], bijector)
    return carry, dist.sample(seed=rng_key)


def sample_and_log_prob_policy(
    rng_key: PRNGKey,
    observations: Array,
    carry: list[Carry],
    params: Dict,
    network: Union[LSTM, GRU],
    bijector: Chain,
) -> tuple[list[Carry], Array, Array]:
    carry, m = network.apply({"params": params}, carry, observations)
    dist = squash_policy(m, params["log_std"], bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return carry, action, log_prob


def log_prob_policy(
    actions: Array,
    observations: Array,
    carry: list[Carry],
    params: Dict,
    network: Union[LSTM, GRU],
    bijector: Chain,
) -> Array:
    _, m = network.apply({"params": params}, carry, observations)
    dist = squash_policy(m, params["log_std"], bijector)
    return dist.log_prob(actions)


def get_recurrent_policy(network: Union[LSTM, GRU], bijector: Chain):
    return RecurrentPolicy(
        dim=network.dim,
        reset=partial(reset_policy, network=network),
        sample=partial(sample_policy, network=network, bijector=bijector),
        log_prob=partial(log_prob_policy, network=network, bijector=bijector),
        sample_and_log_prob=partial(
            sample_and_log_prob_policy, network=network, bijector=bijector
        ),
    )


def log_prob_policy_pathwise(
    policy: RecurrentPolicy, params: Dict, particles: OuterParticles
) -> Array:
    def body(t, log_probs):
        actions = particles.actions[t]
        observations = particles.observations[t - 1]
        carry = jax.tree.map(lambda x: x[t - 1], particles.carry)
        log_prob_incs = policy.log_prob(actions, observations, carry, params)
        return log_probs + log_prob_incs

    num_time_steps, batch_size, _ = particles.actions.shape
    init_log_probs = jnp.zeros(batch_size)
    log_probs = jax.lax.fori_loop(1, num_time_steps, body, init_log_probs)
    return log_probs


@partial(jax.jit, static_argnums=(0,))
def train_step(
    policy: RecurrentPolicy,
    train_state: TrainState,
    particles: OuterParticles,
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        log_probs = log_prob_policy_pathwise(policy, params, particles)
        return -1.0 * jnp.mean(log_probs)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss
