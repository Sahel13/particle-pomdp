from typing import Dict, Callable

import jax
from jax import random
from jax import Array
import jax.numpy as jnp
from chex import PRNGKey

from flax.training.train_state import TrainState

import distrax
from distrax import (
    MultivariateNormalDiag,
    Transformed,
    ScalarAffine,
    Block,
    Chain
)

import optax

from ppomdp.core import (
    LSTMCarry,
    RecurrentPolicy,
    TransitionModel,
    ObservationModel
)
from ppomdp.smc import (
    smc,
    backward_tracing
)
from ppomdp.policy import LSTM


num_outer_particles = 128
num_inner_particles = 64
num_time_steps = 50

dim_state = 1
dim_action = 1
dim_obs = 1


def mean_trans(s: Array, a: Array) -> Array:
    return s + 0.1 * a


def stddev_trans(s: Array, a: Array) -> Array:
    return jnp.array([0.1])


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_trans(s, a),
        scale_diag=stddev_trans(s, a)
    )
    return dist.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_trans(s, a),
        scale_diag=stddev_trans(s, a)
    )
    return dist.log_prob(sn)


def mean_obs(s: Array) -> Array:
    return s


def stddev_obs(s: Array) -> Array:
    b = jnp.array([5.0])  # beacon position
    return 0.5 * (b - s)**2 + 1e-4


def sample_obs(rng_key: Array, s: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_obs(s),
        scale_diag=stddev_obs(s)
    )
    return dist.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    dist = MultivariateNormalDiag(
        loc=mean_obs(s),
        scale_diag=stddev_obs(s)
    )
    return dist.log_prob(z)


def reward_fn(s: Array, a: Array) -> Array:
    Q = jnp.eye(1) * 10.0  # state weights
    R = jnp.eye(1) * 1e-2  # action weights
    reward = - 0.5 * s.T @ Q @ s - 0.5 * a.T @ R @ a
    return jnp.squeeze(reward)


lstm = LSTM(
    dim=dim_action,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
)

bijector = Chain([
    ScalarAffine(0.0, 1.0),
])


def reset_policy(batch_size: int) -> list[LSTMCarry]:
    carry = []
    for _size in lstm.recurr_size:
        mem_shape = (batch_size, _size)
        c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
        carry.append((c, h))
    return carry


def squash_policy(
    a: Array, log_std: Array,
) -> Transformed:

    raw = MultivariateNormalDiag(
        loc=a, scale_diag=jnp.exp(log_std)
    )
    squashed = Transformed(
        distribution=raw,
        bijector=Block(bijector, ndims=1)
    )
    return squashed


def sample_policy(
    rng_key: PRNGKey, s: Array, carry: list[LSTMCarry], params: Dict
) -> tuple[list[LSTMCarry], Array]:
    carry, a = lstm.apply({"params": params}, carry, s)
    dist = squash_policy(a, params["log_std"])
    return carry, dist.sample(seed=rng_key)


def log_prob_policy(
    a: Array, s: Array, carry: list[LSTMCarry], params: Dict
) -> Array:
    carry, a = lstm.apply({"params": params}, carry, s)
    dist = squash_policy(a, params["log_std"])
    return dist.log_prob(a)


prior_dist = distrax.MultivariateNormalDiag(
    loc=jnp.ones((dim_state,)),
    scale_diag=jnp.ones((dim_state,))
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
policy = RecurrentPolicy(
    dim=dim_action, reset=reset_policy, sample=sample_policy, log_prob=log_prob_policy
)


def init_training(
    rng_key: PRNGKey,
    optimizer: Callable,
    learning_rate: float
) -> TrainState:
    key, obs_key, param_key = random.split(rng_key, 3)
    init_carry = policy.reset(num_outer_particles)
    init_obs = random.normal(obs_key, (num_outer_particles, dim_obs))
    init_params = lstm.init(param_key, init_carry, init_obs)["params"]
    tx = optimizer(learning_rate)
    return TrainState.create(apply_fn=lstm.apply, params=init_params, tx=tx)


def step_training(
    train_state: TrainState,
    traced_states: tuple[Array, Array, list[LSTMCarry]]
):
    def loss_fn(params):
        def accumulate(carry, args):
            t = args
            obs = traced_states[0][t - 1]                                # observation
            action = traced_states[1][t]                                 # action
            memory = jax.tree.map(lambda x: x[t - 1], traced_states[2])  # carry
            log_prob = log_prob_policy(action, obs, memory, params)
            return log_prob, log_prob

        _, log_prob = jax.lax.scan(
            accumulate,
            jnp.zeros((num_outer_particles,)),
            jnp.arange(1, num_time_steps)
        )
        return -1.0 * jnp.mean(jnp.sum(log_prob, axis=0))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


key = random.PRNGKey(0)
key, sub_key = random.split(key)
train_state = init_training(sub_key, optax.adam, 1e-3)

for _ in range(25):
    # run interleaved smc
    key, sub_key = random.split(key)
    outer_states, inner_states = smc(
        sub_key,
        num_outer_particles,
        num_inner_particles,
        num_time_steps,
        prior_dist,
        trans_model,
        obs_model,
        policy,
        train_state.params,
        reward_fn,
        tempering=0.5,
    )

    # trace ancestors of outer states
    key, sub_key = random.split(key)
    traced_outer_states = backward_tracing(sub_key, outer_states)

    # update policy parameters
    train_state, loss = step_training(train_state, traced_outer_states)
