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
from ppomdp.policy import LSTM, get_recurrent_policy, train_step
from ppomdp.bijector import Tanh
from ppomdp.utils import batch_data

from matplotlib import pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

num_outer_particles = 256
num_inner_particles = 1024
num_time_steps = 25

state_dim = 1
action_dim = 1
obs_dim = 1


def mean_trans(s: Array, a: Array) -> Array:
    return s + 0.1 * a


def stddev_trans(s: Array, a: Array) -> Array:
    return jnp.array([0.01])


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
    return 1e-2 * jnp.ones_like(s) + (b - s)**2 / 2.0


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


lstm = LSTM(
    dim=action_dim,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(jnp.log(jnp.sqrt(1.0))),
)

bijector = Chain([ScalarAffine(0.0, 2.5), Tanh()])

prior_dist = distrax.MultivariateNormalDiag(
    loc=2.0 * jnp.ones((state_dim,)),
    scale_diag=1.25 * jnp.ones((state_dim,))
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
policy = get_recurrent_policy(lstm, bijector)

rng_key = random.PRNGKey(1337)
learning_rate = 3e-4
batch_size = 32
num_epochs = 500
tempering = 0.15

# Initialize training state
key, obs_key, param_key = random.split(rng_key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
init_params = lstm.init(param_key, init_carry, init_obs)["params"]
tx = optax.adam(learning_rate)
train_state = TrainState.create(apply_fn=lstm.apply, params=init_params, tx=tx)

for i in range(1, num_epochs + 1):
    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_info, log_marginal = smc(
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
        backward_tracing(sub_key, outer_states, inner_states, inner_info)

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


train_state.params["log_std"] = -20.0 * jnp.ones((1,))

key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, log_marginal = smc(
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
    backward_tracing(sub_key, outer_states, inner_states, inner_infos)

observations = outer_states.particles[0]
actions = outer_states.particles[1]
state_means = inner_infos.mean
state_covars = inner_infos.covar
rwrds = outer_states.rewards

# indices = jnp.where((state_means >= 2.25) & (state_means <= 2.75))[0]

fig, axs = plt.subplots(4, 1, figsize=(10, 8))
for n in range(num_outer_particles):
    axs[0].plot(actions[:, n, :])
    axs[0].set_title('Action over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Action')

    axs[1].plot(observations[:, n, :])
    axs[1].set_title('Observation over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Observation')

    axs[2].plot(state_means[:, n, :])
    axs[2].set_title('Mean over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('State')

    axs[3].plot(state_covars[:, n, 0, 0])
    axs[3].set_title('Variance over Time')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Variance')

plt.tight_layout()
plt.show()


train_state.params["log_std"] = -20.0 * jnp.ones((1,))

states = []
actions = []
observations = []

# key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

init_dist = distrax.MultivariateNormalDiag(
    loc=1.0 * jnp.ones((state_dim,)),
    scale_diag=0.25 * jnp.ones((state_dim,))
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

# Convert lists to arrays for plotting
states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))
observations = jnp.squeeze(jnp.array(observations))

plt.plot(states)
plt.plot(actions)
plt.show()
