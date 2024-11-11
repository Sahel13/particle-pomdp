from typing import Callable

import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from distrax import Chain, MultivariateNormalDiag, ScalarAffine
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import Array, random
from matplotlib import pyplot as plt
from functools import partial

from ppomdp.bijector import Tanh
from ppomdp.core import ObservationModel, TransitionModel
from ppomdp.policy import LSTM, get_recurrent_policy, train_step
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data

jax.config.update("jax_enable_x64", True)


def euler_step(deriv_fn: Callable, s: Array, a: Array, dt: float) -> Array:
    return s + deriv_fn(s, a) * dt


def pendulum_ode(s: Array, a: Array) -> Array:
    m, l = 1.0, 1.0
    g, d = 9.81, 1e-3

    q, dq = s[0], s[1]
    ddq = -3.0 * g / (2.0 * l) * jnp.sin(q)
    ddq += (a[0] - d * dq) * 3.0 / (m * l**2)
    return jnp.array([dq, ddq])


trans_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.array([1e-4, 0.025]))
obs_noise = MultivariateNormalDiag(jnp.zeros(2), scale_diag=jnp.ones(2) * 1e-4)


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    return euler_step(pendulum_ode, s, a, 0.05) + trans_noise.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    mean = euler_step(pendulum_ode, s, a, 0.05)
    return trans_noise.log_prob(sn - mean)


def get_obs(s: Array) -> Array:
    """Position in Cartesian coordinates."""
    return jnp.array((jnp.sin(s[0]), jnp.cos(s[0])))


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    return get_obs(s) + obs_noise.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    mean_obs = get_obs(s)
    return obs_noise.log_prob(z - mean_obs)


def reward_fn(s: Array, a: Array) -> Array:
    Q = jnp.array([10.0, 1.0])
    R = 1e-2
    goal = jnp.array([jnp.pi, 0.0])

    def wrap_angle(q: Array) -> Array:
        return jnp.array((q[0] % (2 * jnp.pi), q[1]))

    s = wrap_angle(s)
    cost = jnp.dot((s - goal) * Q, (s - goal)) + R * jnp.dot(a, a)
    return -0.5 * cost


state_dim = 2
action_dim = 1
obs_dim = 2

num_outer_particles = 256
num_inner_particles = 256
num_time_steps = 100

lstm = LSTM(
    dim=action_dim,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(jnp.log(jnp.sqrt(1.0))),
)
bijector = Chain([ScalarAffine(0.0, 2.5), Tanh()])


prior_dist = MultivariateNormalDiag(
    loc=jnp.zeros((state_dim,)), scale_diag=jnp.ones((state_dim,)) * 1e-16
)
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)
policy = get_recurrent_policy(lstm, bijector)

rng_key = random.PRNGKey(77)
learning_rate = 1e-3
batch_size = 32
num_epochs = 25
tempering = 0.2

# Initialize training state
key, obs_key, param_key = random.split(rng_key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
init_params = lstm.init(param_key, init_carry, init_obs)["params"]
tx = optax.adam(learning_rate)
train_state = TrainState.create(apply_fn=lstm.apply, params=init_params, tx=tx)

# Run SMC and plot smoothed trajectories.
key, sub_key = random.split(key)
outer_states, inner_states, _ = smc(
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
    tempering=tempering,
)

# trace ancestors of outer states
key, sub_key = random.split(key)
traced_outer_states, traced_inner_states = backward_tracing(sub_key, outer_states, inner_states)

@partial(jnp.vectorize, signature="(n,d),(n)->(d)")
def compute_empirical_mean(particles: Array, weights: Array) -> Array:
    return jnp.sum(particles * weights[:, None], axis=0)

key, sub_key = random.split(key)
random_idx = random.choice(sub_key, num_outer_particles)
mean_particles = compute_empirical_mean(traced_inner_states.particles, traced_inner_states.weights)
angles = mean_particles[:, :, 0]
angular_velocities = mean_particles[:, :, 1]
actions = traced_outer_states.particles.actions[:, :, 0]

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
# Set a title for the plots
fig.suptitle("Smoothed trajectories")
axs[0].plot(angles, label="angle")
axs[0].set_ylabel("Angle")
axs[0].grid(True)
axs[1].plot(angular_velocities, label="angular velocity")
axs[1].set_ylabel("Angular velocity")
axs[1].grid(True)
axs[2].plot(actions, label="action")
axs[2].set_ylabel("Action")
axs[2].set_xlabel("Time step")
axs[2].grid(True)
plt.tight_layout()
plt.show()

# The training loop
for i in range(1, num_epochs + 1):
    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, log_marginal = smc(
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
        tempering=tempering,
    )

    # trace ancestors of outer states
    key, sub_key = random.split(key)
    traced_outer, _ = backward_tracing(sub_key, outer_states, inner_states)

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_outer_particles, batch_size)
    for batch_idx in batch_indices:
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer.particles)
        train_state, batch_loss = train_step(policy, train_state, outer_batch)
        loss += batch_loss

    print(f"Iter: {i}, Loss: {loss}, Log marginal: {log_marginal}")


train_state.params["log_std"] = -20.0 * jnp.ones((1,))

states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

state = jnp.array([0.0, 0.0])
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

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(states[:, 0])
axs[0].set_ylabel("Angle")
axs[0].grid(True)
axs[1].plot(states[:, 1])
axs[1].set_ylabel("Angular Velocity")
axs[1].grid(True)
axs[2].plot(actions)
axs[2].set_ylabel("Action")
axs[2].grid(True)
plt.tight_layout()
plt.show()
