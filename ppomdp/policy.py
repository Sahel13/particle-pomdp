from functools import partial
from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Block, Chain, MultivariateNormalDiag, Transformed
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import Array

from ppomdp.core import LSTMCarry, RecurrentPolicy


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
        init_kernel (Callable): Initializer for the kernel weights.
    """

    dim: int
    feature_fn: Callable
    encoder_size: Sequence[int]
    recurr_size: Sequence[int]
    output_size: Sequence[int]
    init_log_std: Callable = nn.initializers.ones
    init_kernel: Callable = nn.initializers.he_uniform()

    @nn.compact
    def __call__(
        self, carry: list[LSTMCarry], s: Array
    ) -> tuple[list[LSTMCarry], Array]:
        log_std = self.param("log_std", self.init_log_std, self.dim)

        # pass inputs through features layer
        y = self.feature_fn(s)

        # pass features through encoding layers
        for _size in self.encoder_size:
            y = nn.relu(nn.Dense(_size)(y))
        y = nn.Dense(self.recurr_size[0])(y)

        # pass encodings through recurrent layers
        for k, _size in enumerate(self.recurr_size):
            carry[k], y = nn.LSTMCell(_size)(carry[k], y)

        # pass result through output layers
        for _size in self.output_size:
            y = nn.relu(nn.Dense(_size)(y))
        a = nn.Dense(self.dim)(y)
        return carry, a


def reset_policy(batch_size: int, lstm: LSTM) -> list[LSTMCarry]:
    carry = []
    for _size in lstm.recurr_size:
        mem_shape = (batch_size, _size)
        c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
        carry.append((c, h))
    return carry


def squash_policy(bijector: Chain, actions: Array, log_std: Array) -> Transformed:
    raw = MultivariateNormalDiag(loc=actions, scale_diag=jnp.exp(log_std))
    squashed = Transformed(distribution=raw, bijector=Block(bijector, ndims=1))
    return squashed


def sample_policy(
    rng_key: PRNGKey,
    observations: Array,
    carry: list[LSTMCarry],
    params: Dict,
    lstm: LSTM,
    bijector: Chain,
) -> tuple[list[LSTMCarry], Array]:
    carry, mean_actions = lstm.apply({"params": params}, carry, observations)
    dist = squash_policy(bijector, mean_actions, params["log_std"])
    return carry, dist.sample(seed=rng_key)


def log_prob_policy(
    actions: Array,
    observations: Array,
    carry: list[LSTMCarry],
    params: Dict,
    lstm: LSTM,
    bijector: Chain,
) -> Array:
    carry, mean_actions = lstm.apply({"params": params}, carry, observations)
    dist = squash_policy(bijector, mean_actions, params["log_std"])
    return dist.log_prob(actions)


def get_recurrent_policy(lstm: LSTM, bijector: Chain):
    return RecurrentPolicy(
        dim=lstm.dim,
        reset=partial(reset_policy, lstm=lstm),
        sample=partial(sample_policy, lstm=lstm, bijector=bijector),
        log_prob=partial(log_prob_policy, lstm=lstm, bijector=bijector),
    )


@partial(jax.jit, static_argnums=(1,))
def train_step(
    train_state: TrainState,
    policy: RecurrentPolicy,
    traced_states: tuple[Array, Array, list[LSTMCarry]],
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        batch_size = traced_states[0].shape[1]
        num_time_steps = traced_states[0].shape[0] - 1

        def accumulate(_, t):
            observations = traced_states[0][t]
            actions = traced_states[1][t]
            carry = jax.lax.cond(
                t == 0,
                lambda _: policy.reset(batch_size),
                lambda _: jax.tree.map(lambda x: x[t - 1], traced_states[2]),
                t,
            )
            log_prob = policy.log_prob(actions, observations, carry, params)
            return None, log_prob

        _, log_prob = jax.lax.scan(accumulate, None, jnp.arange(num_time_steps))
        return -1.0 * jnp.mean(jnp.sum(log_prob, axis=0))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss
