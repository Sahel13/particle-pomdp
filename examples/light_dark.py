from typing import Dict, Callable

import jax
from jax import Array, random
import jax.numpy as jnp

from chex import PRNGKey
from flax.training.train_state import TrainState
from flax.linen.initializers import constant

import optax

import distrax
from distrax import (
    MultivariateNormalDiag,
    Transformed,
    ScalarAffine,
    Block,
    Chain
)

from ppomdp.core import (
    LSTMCarry,
    RecurrentPolicy,
    TransitionModel,
    ObservationModel, OuterParticles
)
from ppomdp.smc import (
    smc,
    backward_tracing
)
from ppomdp.policy import LSTM
from ppomdp.bijector import Tanh, Sigmoid
from ppomdp.utils import batch_data

from matplotlib import pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

num_outer_particles = 256
num_inner_particles = 64
num_time_steps = 25

dim_state = 1
dim_action = 1
dim_obs = 1


def mean_trans(s: Array, a: Array) -> Array:
    return s + 0.1 * a


def stddev_trans(s: Array, a: Array) -> Array:
    return jnp.array([1e-2])


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
    return jnp.sqrt(jnp.abs(b - s)) + 1e-4


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


def reward_fn(s: Array, q: Array, a: Array, t: int) -> Array:
    Q = jax.lax.cond(
        t < num_time_steps,
        lambda _: jnp.eye(1) * 0.1,
        lambda _: jnp.eye(1) * 200.0,
        operand=None
    )
    # jax.debug.print("t: {}", t)
    # jax.debug.print("Q: {}", Q)
    R = jnp.eye(1) * 0.0
    g = jnp.array([0.0])

    reward = (
        - 0.5 * (s - g).T @ Q @ (s - g) - 0.5 * a.T @ R @ a
        # - 0.5 * 0.25 * jnp.dot(a - q, a - q)
    )
    return jnp.squeeze(reward)


lstm = LSTM(
    dim=dim_action,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(0.5)
)

bijector = Chain([
    ScalarAffine(0.0, 0.1), # Sigmoid()
])


def reset_policy(batch_size: int) -> list[LSTMCarry]:
    carry = []
    for _size in lstm.recurr_size:
        mem_shape = (batch_size, _size)
        c, h = jnp.zeros(mem_shape), jnp.zeros(mem_shape)  # LSTMCarry
        carry.append((c, h))
    return carry


def squash_policy(
    m: Array, log_std: Array,
) -> Transformed:
    raw = MultivariateNormalDiag(
        loc=m, scale_diag=jnp.exp(log_std)
    )
    squashed = Transformed(
        distribution=raw,
        bijector=Block(bijector, ndims=1)
    )
    return squashed


def sample_policy(
    rng_key: PRNGKey, s: Array, carry: list[LSTMCarry], params: Dict
) -> tuple[list[LSTMCarry], Array]:
    carry, m = lstm.apply({"params": params}, carry, s)
    dist = squash_policy(m, params["log_std"])
    return carry, dist.sample(seed=rng_key)


def log_prob_policy(
    a: Array, s: Array, carry: list[LSTMCarry], params: Dict
) -> Array:
    carry, m = lstm.apply({"params": params}, carry, s)
    dist = squash_policy(m, params["log_std"])
    return dist.log_prob(a)


prior_dist = distrax.MultivariateNormalDiag(
    loc=2.5 * jnp.ones((dim_state,)),
    scale_diag=jnp.sqrt(5.0) * jnp.ones((dim_state,))
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
    traced_states: OuterParticles,
    batch_size: int,
):
    def loss_fn(params):
        def accumulate(carry, args):
            t = args
            obs = traced_states.observations[t]
            action = traced_states.actions[t]
            memory = jax.tree.map(lambda x: x[t - 1], traced_states.carry)
            log_prob = log_prob_policy(action, obs, memory, params)
            return log_prob, log_prob

        obs = traced_states.observations[0]
        action = traced_states.actions[0]
        memory = policy.reset(batch_size)
        init_log_prob = log_prob_policy(obs, action, memory, params)

        _, log_prob = jax.lax.scan(
            accumulate,
            init_log_prob,
            jnp.arange(1, num_time_steps - 1)
        )
        return -1.0 * jnp.mean(jnp.sum(log_prob, axis=0))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


key = random.PRNGKey(0)
key, sub_key = random.split(key)
train_state = init_training(sub_key, optax.adam, 1e-3)

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
    tempering=0.25,
    resample=True
)

# trace ancestors of outer states
key, sub_key = random.split(key)
traced_outer, traced_inner = \
    backward_tracing(sub_key, outer_states, inner_states)

observations = traced_outer.particles[0]
actions = traced_outer.particles[1]
states = jnp.mean(traced_inner.particles, axis=2)
# states = traced_inner.particles

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
for n in range(0, 10):
    axs[0].plot(actions[:, n, :])
    axs[0].set_title('Action over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Action')

    axs[1].plot(observations[:, n, :])
    axs[1].set_title('Observation over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Observation')

    # for m in range(num_inner_particles):
    # axs[2].plot(states[:, n, m, :])
    axs[2].plot(states[:, n, :])
    axs[2].set_title('State over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('State')

plt.tight_layout()
plt.show()


# from copy import deepcopy
#
# key = random.PRNGKey(0)
# key, sub_key = random.split(key)
# train_state = init_training(sub_key, optax.adam, 5e-4)
#
# batch_size = 64
# for i in range(100):
#     # evaluate mean policy
#     eval_state = deepcopy(train_state)
#     eval_state.params["log_std"] = -20.0 * jnp.ones((dim_action,))
#
#     key, sub_key = random.split(key)
#     outer_states, _ = smc(
#         sub_key,
#         num_outer_particles,
#         num_inner_particles,
#         num_time_steps,
#         prior_dist,
#         trans_model,
#         obs_model,
#         policy,
#         eval_state.params,
#         reward_fn,
#         tempering=0.0,
#         resample=False,
#     )
#     cumulative_return = jnp.mean(jnp.sum(outer_states.rewards, axis=0))
#
#     # run interleaved smc
#     key, sub_key = random.split(key)
#     outer_states, inner_states = smc(
#         sub_key,
#         num_outer_particles,
#         num_inner_particles,
#         num_time_steps,
#         prior_dist,
#         trans_model,
#         obs_model,
#         policy,
#         train_state.params,
#         reward_fn,
#         tempering=0.1,
#     )
#
#     # trace ancestors of outer states
#     key, sub_key = random.split(key)
#     traced_outer, _ = backward_tracing(sub_key, outer_states, inner_states)
#
#     # update policy parameters
#     loss = 0.0
#     key, sub_key = random.split(key)
#     batch_indicies = batch_data(sub_key, num_outer_particles, batch_size)
#     for batch_idx in batch_indicies:
#         outer_batch = jax.tree.map(lambda x: x[:, batch_idx, :], traced_outer.particles)
#         train_state, batch_loss = step_training(train_state, outer_batch, batch_size)
#         loss += batch_loss
#
#     print(f"Iter: {i}, Loss: {loss}, Return: {cumulative_return}")
#
#
# key = random.PRNGKey(0)
#
# train_state.params["log_std"] = -20.0 * jnp.ones((1,))
#
# states = []
# actions = []
# observations = []
#
# key, state_key, obs_key = random.split(key, 3)
# state = jnp.array([[2.5]])  # prior_dist.sample(seed=state_key, sample_shape=(1,))
# obs = sample_obs(obs_key, state)
# carry = policy.reset(1)
#
# states.append(state)
# observations.append(obs)
#
# for _ in range(num_time_steps):
#     key, state_key, obs_key, action_key = random.split(key, 4)
#     carry, action = policy.sample(action_key, obs, carry, train_state.params)
#     state = sample_trans(state_key, state, action)
#     obs = sample_obs(obs_key, state)
#
#     states.append(state)
#     actions.append(action)
#     observations.append(obs)
#
#
# # Convert lists to arrays for plotting
# states = jnp.squeeze(jnp.array(states))
# actions = jnp.squeeze(jnp.array(actions))
# observations = jnp.squeeze(jnp.array(observations))
#
# # Plotting
# fig, axs = plt.subplots(3, 1, figsize=(10, 8))
#
# axs[0].plot(states)
# axs[0].set_title('State over Time')
# axs[0].set_xlabel('Time Step')
# axs[0].set_ylabel('State')
#
# axs[1].plot(actions)
# axs[1].set_title('Action over Time')
# axs[1].set_xlabel('Time Step')
# axs[1].set_ylabel('Action')
#
# axs[2].plot(observations)
# axs[2].set_title('Observation over Time')
# axs[2].set_xlabel('Time Step')
# axs[2].set_ylabel('Observation')
#
# plt.tight_layout()
# plt.show()
#
#
# # def f(x):
# #     return -0.5 * x**2 * 10.0
# #
# # key = random.PRNGKey(0)
# # key, sub_key = random.split(key)
# # x = random.normal(sub_key, (1000,)) * 0.1
# #
# # m = jnp.mean(f(x))
# # print("expected value: ", m)
