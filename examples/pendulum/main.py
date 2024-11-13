import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from distrax import Chain, MultivariateNormalDiag, ScalarAffine
from environment import obs_model, reward_fn, trans_model
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import random

from ppomdp.bijector import Tanh
from ppomdp.policy import LSTM, get_recurrent_policy, train_step
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data, weighted_mean

jax.config.update("jax_enable_x64", True)

state_dim = 2
action_dim = 1
obs_dim = 2

lstm = LSTM(
    dim=action_dim,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(jnp.log(1.0)),
)
bijector = Chain([ScalarAffine(0.0, 2.5), Tanh()])
prior_dist = MultivariateNormalDiag(
    loc=jnp.zeros((state_dim,)), scale_diag=jnp.ones((state_dim,)) * 1e-16
)
policy = get_recurrent_policy(lstm, bijector)

rng_key = random.PRNGKey(10)
learning_rate = 1e-3
batch_size = 32
num_epochs = 400
tempering = 0.1
num_outer_particles = 256
num_inner_particles = 256
num_time_steps = 100

# Initialize training state
key, obs_key, param_key = random.split(rng_key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
init_params = lstm.init(param_key, init_carry, init_obs)["params"]
scheduler = optax.constant_schedule(learning_rate)
tx = optax.adam(scheduler)
train_state = TrainState.create(apply_fn=lstm.apply, params=init_params, tx=tx)

jitted_smc = jax.jit(smc, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9, 10))
jitted_backward_tracing = jax.jit(backward_tracing, static_argnums=(5))

# Run SMC and plot smoothed trajectories.
key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, _ = jitted_smc(
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
    slew_rate_penalty=0.05,
)

key, sub_key = random.split(key)
traced_outer_states, traced_inner_states, _ = jitted_backward_tracing(
    sub_key, outer_states, inner_states, inner_infos
)

mean_particles = weighted_mean(
    traced_inner_states.particles, traced_inner_states.weights
)
angles = mean_particles[:, :, 0]
angular_velocities = mean_particles[:, :, 1]
actions = traced_outer_states.particles.actions[:, :, 0]

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
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
    start_time = time.time()
    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_infos, log_marginal = jitted_smc(
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
    traced_outer, traced_inner, _ = jitted_backward_tracing(
        sub_key, outer_states, inner_states, inner_infos
    )

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_outer_particles, batch_size)
    for batch_idx in batch_indices:
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer.particles)
        train_state, batch_loss = train_step(policy, train_state, outer_batch)
        loss += batch_loss

    log_std = train_state.params["log_std"][0]
    end_time = time.time()
    time_diff = end_time - start_time
    if i % 5 == 0:
        print(
            f"Epoch: {i:3d}, Log marginal: {log_marginal:8.3f}, Log std: {log_std:8.3f}, Time per epoch: {time_diff:6.3f}s"
        )

    if i % 100 == 0:
        # Monitor the smoothed trajectories every 100 epochs.
        mean_particles = weighted_mean(traced_inner.particles, traced_inner.weights)
        angles = mean_particles[:, :, 0]
        angular_velocities = mean_particles[:, :, 1]
        actions = traced_outer.particles.actions[:, :, 0]

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
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

train_state.params["log_std"] = -20.0 * jnp.ones((1,))

states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

state = jnp.array([0.0, 0.0])
obs = obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, obs, carry, train_state.params)
    state = trans_model.sample(state_key, state, action[0])
    obs = obs_model.sample(obs_key, state)

    states.append(state)
    actions.append(action[0])
    observations.append(obs)

# Convert lists to arrays for plotting
states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))
observations = jnp.squeeze(jnp.array(observations))

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Simulated trajectories")

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
