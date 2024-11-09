from functools import partial
from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Block, Chain, MultivariateNormalDiag, Transformed
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import Array

from ppomdp.core import LSTMCarry, OuterParticles, RecurrentPolicy


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
    def __call__(
        self, carry: list[LSTMCarry], s: Array
    ) -> tuple[list[LSTMCarry], Array]:
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


def reset_policy(batch_size: int, lstm: LSTM) -> list[LSTMCarry]:
    carry = []
    for _size in lstm.recurr_size:
        mem_shape = (batch_size, _size)
        c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
        carry.append((c, h))
    return carry


def squash_policy(mean: Array, log_std: Array, bijector: Chain) -> Transformed:
    dist = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    return Transformed(distribution=dist, bijector=Block(bijector, ndims=1))


def sample_policy(
    rng_key: PRNGKey,
    observations: Array,
    carry: list[LSTMCarry],
    params: Dict,
    lstm: LSTM,
    bijector: Chain,
) -> tuple[list[LSTMCarry], Array]:
    carry, m = lstm.apply({"params": params}, carry, observations)
    dist = squash_policy(m, params["log_std"], bijector)
    return carry, dist.sample(seed=rng_key)


def sample_and_log_prob_policy(
    rng_key: PRNGKey,
    observations: Array,
    carry: list[LSTMCarry],
    params: Dict,
    lstm: LSTM,
    bijector: Chain,
) -> tuple[list[LSTMCarry], Array, Array]:
    carry, m = lstm.apply({"params": params}, carry, observations)
    dist = squash_policy(m, params["log_std"], bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return carry, action, log_prob


def log_prob_policy(
    actions: Array,
    observations: Array,
    carry: list[LSTMCarry],
    params: Dict,
    lstm: LSTM,
    bijector: Chain,
) -> Array:
    _, m = lstm.apply({"params": params}, carry, observations)
    dist = squash_policy(m, params["log_std"], bijector)
    return dist.log_prob(actions)


def get_recurrent_policy(lstm: LSTM, bijector: Chain):
    return RecurrentPolicy(
        dim=lstm.dim,
        reset=partial(reset_policy, lstm=lstm),
        sample=partial(sample_policy, lstm=lstm, bijector=bijector),
        log_prob=partial(log_prob_policy, lstm=lstm, bijector=bijector),
        sample_and_log_prob=partial(
            sample_and_log_prob_policy, lstm=lstm, bijector=bijector
        ),
    )


def log_prob_policy_pathwise(
    policy: RecurrentPolicy, params: Dict, particles: OuterParticles
):
    def body(log_prob, t):
        actions = particles.actions[t]
        observations = particles.observations[t]
        carry = jax.tree.map(lambda x: x[t - 1], particles.carry)
        log_prob_inc = policy.log_prob(actions, observations, carry, params)
        return log_prob + log_prob_inc, log_prob_inc

    num_time_steps, batch_size, _ = particles.actions.shape

    init_actions = particles.actions[0]
    init_observations = particles.observations[0]
    init_carry = policy.reset(batch_size)
    init_log_prob = policy.log_prob(init_actions, init_observations, init_carry, params)

    _, log_prob_inc = jax.lax.scan(
        body, init_log_prob, jnp.arange(1, num_time_steps - 1)
    )
    return jnp.vstack((init_log_prob, log_prob_inc))


@partial(jax.jit, static_argnums=(0,))
def train_step(
    policy: RecurrentPolicy,
    train_state: TrainState,
    particles: OuterParticles,
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        log_probs = log_prob_policy_pathwise(policy, params, particles)
        return -1.0 * jnp.mean(jnp.sum(log_probs, axis=0))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss
