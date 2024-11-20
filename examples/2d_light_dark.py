from typing import Callable

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
    ScalarAffine,
    Chain
)
from ppomdp.core import (
    TransitionModel,
    ObservationModel
)
from ppomdp.smc import (
    smc,
    backward_tracing
)
from ppomdp.policy import (
    LSTM,
    GRU,
    get_recurrent_policy,
    train_step
)
from ppomdp.bijector import Tanh
from ppomdp.utils import batch_data

from matplotlib import pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

num_outer_particles = 256
num_inner_particles = 256
num_time_steps = 50

state_dim = 2
action_dim = 2
obs_dim = 2


def mean_trans(s: Array, a: Array) -> Array:
    return s + 0.1 * a


def stddev_trans(s: Array, a: Array) -> Array:
    return jnp.array([1e-2, 1e-2])


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
    b = jnp.array([5.0, 0.0])  # beacon position
    H = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    return 1e-2 * jnp.ones(obs_dim) + (b - H @ s)**2 / 2.0


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


def reward_fn(s: Array, a: Array, t: int) -> Array:
    state_cost = jax.lax.cond(
        t < num_time_steps - 1,
        lambda _: 0.0,
        lambda _: 150. * jnp.dot(s, s),
        operand=None
    )
    action_cost = 1e-3 * jnp.dot(a, a)
    return - 0.5 * state_cost - 0.5 * action_cost


#
init_log_std = jnp.log(jnp.array([1.0, 1.0]))

network = GRU(
    dim=action_dim,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(init_log_std),
)

shift = jnp.zeros((action_dim,))
scale = jnp.array([2.0, 2.0])
bijector = Chain([ScalarAffine(shift, scale), Tanh()])

prior_dist = distrax.MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0]),
    scale_diag=jnp.array([2.5, 1e-2])
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
policy = get_recurrent_policy(network, bijector)

rng_key = random.PRNGKey(1337)
learning_rate = 3e-4
batch_size = 64
num_epochs = 500
tempering = 0.05

# Initialize training state
key, obs_key, param_key = random.split(rng_key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
init_params = network.init(param_key, init_carry, init_obs)["params"]
train_state = TrainState.create(
    apply_fn=network.apply,
    params=init_params,
    tx=optax.adam(learning_rate)
)

jitted_smc = jax.jit(smc, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9, 10))
jitted_backward_tracing = jax.jit(backward_tracing, static_argnums=(5,))

#
for i in range(1, num_epochs + 1):
    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_info, log_marginal = \
        jitted_smc(
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
            tempering,
        )

    # trace ancestors of outer states
    key, sub_key = random.split(key)
    traced_outer, traced_inner, _ = \
        jitted_backward_tracing(sub_key, outer_states, inner_states, inner_info)

    variance = jnp.mean(jnp.var(traced_inner.particles[-1], axis=1), axis=0)

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_outer_particles, batch_size)
    for batch_idx in batch_indices:
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer.particles)
        train_state, batch_loss = train_step(policy, train_state, outer_batch)
        loss += batch_loss

    print(f"Iter: {i}, Loss: {loss}, Log marginal: {log_marginal}, Variance: {variance}")


# train_state.params["log_std"] = -20.0 * jnp.ones((action_dim,))

key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, log_marginal = \
    jitted_smc(
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
        tempering,
    )

# trace ancestors of outer states
key, sub_key = random.split(key)
outer_states, inner_states, inner_infos = \
    jitted_backward_tracing(sub_key, outer_states, inner_states, inner_infos)

#
observations = outer_states.particles[0]
actions = outer_states.particles[1]
state_means = inner_infos.mean
state_covars = inner_infos.covar
rwrds = outer_states.rewards

fig, axs = plt.subplots(6, 1, figsize=(8, 8))
for n in range(num_outer_particles):
    axs[0].plot(actions[:, n, 0])
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Act-1')

    axs[1].plot(actions[:, n, 1])
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Act-2')

    axs[2].plot(state_means[:, n, 0])
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('State-1')

    axs[3].plot(state_means[:, n, 1])
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('State-2')

    axs[4].plot(state_covars[:, n, 0, 0])
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Var-1')

    axs[5].plot(state_covars[:, n, 1, 1])
    axs[5].set_xlabel('Time Step')
    axs[5].set_ylabel('Var-2')

plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 8))
# for n in range(num_outer_particles):
#     plt.plot(state_means[:, n, 0], state_means[:, n, 1])
#
# # points = jnp.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
# # plt.scatter(points[:, 0], points[:, 1], color='red', marker='o', s=100)
#
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()

#
train_state.params["log_std"] = -20.0 * jnp.ones((action_dim,))

states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)
init_dist = distrax.MultivariateNormalDiag(
    loc=jnp.array([2.0, 2.0]),
    scale_diag=jnp.array([1e-2, 1e-2])
)

state = init_dist.sample(seed=state_key)
obs = sample_obs(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, obs, carry, train_state.params)
    state = sample_trans(state_key, state, action[0])
    obs = sample_obs(obs_key, state)

    states.append(state)
    actions.append(action[0])
    observations.append(obs)


states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))
observations = jnp.squeeze(jnp.array(observations))

plt.figure(figsize=(8, 8))
plt.plot(states[:, 0], states[:, 1])

plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
