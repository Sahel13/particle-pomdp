from typing import Dict, Callable
from functools import partial

import jax
from jax import random
from jax import Array
import jax.numpy as jnp
from chex import PRNGKey

from flax.training.train_state import TrainState

from distrax import MultivariateNormalDiag
from distrax import Chain, ScalarAffine

import optax

from ppomdp.core import LSTMCarry
from ppomdp.utils import (
    LSTM,
    initialize_carry,
    lstm_distribution
)


def transition_mean(s: Array, a: Array) -> Array:
    return s + 0.1 * a


def transition_stddev(s: Array, a: Array) -> Array:
    return jnp.array([0.1])


def transition_sample(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=transition_mean(s, a),
        scale_diag=transition_stddev(s, a)
    )
    return dist.sample(seed=rng_key)


def transition_logpdf(sn: Array, s: Array, a: Array) -> float:
    dist = MultivariateNormalDiag(
        loc=transition_mean(s, a),
        scale_diag=transition_stddev(s, a)
    )
    return dist.log_prob(sn)


def observation_mean(s: Array) -> Array:
    return s


def observation_stddev(s: Array) -> Array:
    b = jnp.array([5.0])  # beacon position
    return 0.5 * (b - s)**2


def observation_sample(rng_key: Array, s: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=observation_mean(s),
        scale_diag=observation_stddev(s)
    )
    return dist.sample(seed=rng_key)


def observation_logpdf(z: Array, s: Array) -> float:
    dist = MultivariateNormalDiag(
        loc=observation_mean(s),
        scale_diag=observation_stddev(s)
    )
    return dist.log_prob(z)


def reward_fn(s: Array, a: Array) -> float:
    Q = jnp.eye(1) * 10.0  # state weights
    R = jnp.eye(1) * 1e-2  # action weights
    reward = - 0.5 * s.T @ Q @ s - 0.5 * a.T @ R @ a
    return reward.item()


@partial(jnp.vectorize, signature='(k)->(h)')
def unit_features(s: Array) -> Array:
    return s


lstm = LSTM(
    dim=1,
    feature_fn=unit_features,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
)

bijector = Chain([
    ScalarAffine(0.0, 1.0),
])


def sample_policy(
    rng_key: PRNGKey, s: Array, carry: list[LSTMCarry], params: Dict
) -> tuple[list[LSTMCarry], Array]:
    carry, dist = lstm_distribution(s, carry, lstm, params, bijector)
    return carry, dist.sample(seed=rng_key)


def log_prob_policy(
    a: Array, s: Array, carry: list[LSTMCarry], params: Dict
) -> float:
    _, dist = lstm_distribution(s, carry, lstm, params, bijector)
    return dist.log_prob(a)


def create_train_state(
    rng_key: PRNGKey, module: LSTM, optimizer: Callable, learning_rate: float
) -> TrainState:
    init_key, param_key = random.split(rng_key)
    init_data = random.normal(init_key, (1,))
    init_carry = initialize_carry(module, 1)
    params = module.init(param_key, init_carry, init_data)["params"]
    tx = optimizer(learning_rate)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=1)
def train_step(state: TrainState, module: LSTM, data: tuple[Array, Array]):
    def loss_fn(params):
        apply_fn = partial(state.apply_fn, {"params": params})
        init_carry = initialize_carry(module, 1)
        _, predictions = jax.lax.scan(apply_fn, init_carry, data[0])
        return optax.squared_error(predictions, data[1]).mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
