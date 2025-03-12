import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from distrax import Chain, ScalarAffine
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import random

from ppomdp.bijector import Tanh
from ppomdp.envs import TargetInterceptionEnv
from ppomdp.policy import LSTM, get_recurrent_policy, train_step
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data, weighted_mean

env = TargetInterceptionEnv
lstm = LSTM(
    dim=env.action_dim,
    feature_fn=env.feature_fn,
    encoder_size=[256, 256],
    recurr_size=[32, 32],
    output_size=[256, 256],
    init_log_std=constant(jnp.log(1.0)),
)
bijector = Chain([ScalarAffine(0.0, env.action_scale), Tanh()])
policy = get_recurrent_policy(lstm, bijector)

key = random.key(123)
learning_rate = 1e-3
batch_size = 32
num_epochs = 500
tempering = 0.1
slew_rate_penalty = 0.0
num_outer_particles = 512
num_inner_particles = 256

# Initialize training state
key, obs_key, param_key = random.split(key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, env.obs_dim))
init_params = lstm.init(param_key, init_carry, init_obs)["params"]
scheduler = optax.constant_schedule(learning_rate)
tx = optax.adam(scheduler)
train_state = TrainState.create(apply_fn=lstm.apply, params=init_params, tx=tx)

jitted_smc = jax.jit(smc, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9, 10))
jitted_backward_tracing = jax.jit(backward_tracing, static_argnums=(5))

# Run SMC and plot smoothed trajectories.
key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, log_marginal = jitted_smc(
    sub_key,
    env.num_time_steps,
    num_outer_particles,
    num_inner_particles,
    env.prior_dist,
    env.trans_model,
    env.obs_model,
    policy,
    train_state.params,
    env.reward_fn,
    tempering,
    slew_rate_penalty,
)
print(f"Log marginal: {log_marginal:8.3f}")

key, sub_key = random.split(key)
traced_outer_states, traced_inner_states, _ = jitted_backward_tracing(
    sub_key, outer_states, inner_states, inner_infos
)

mean_particles = weighted_mean(
    traced_inner_states.particles, traced_inner_states.weights
)
actions = traced_outer_states.actions[:, :, 0]

plt.figure()
plt.plot(mean_particles[:, :, 0], mean_particles[:, :, 2])
plt.plot([0], [0], "o", color="orange", markersize=10)
plt.title("Smoothed trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("square")
plt.grid(True)
plt.show()

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()
    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_infos, log_marginal = jitted_smc(
        sub_key,
        env.num_time_steps,
        num_outer_particles,
        num_inner_particles,
        env.prior_dist,
        env.trans_model,
        env.obs_model,
        policy,
        train_state.params,
        env.reward_fn,
        tempering,
        slew_rate_penalty,
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
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer)
        train_state, batch_loss = train_step(policy, train_state, outer_batch)
        loss += batch_loss

    log_std = train_state.params["log_std"][0]
    end_time = time.time()
    time_diff = end_time - start_time

    if i % 10 == 0:
        print(
            f"Epoch: {i:3d}, Log marginal: {log_marginal:8.3f}, Log std: {log_std:8.3f}, Time per epoch: {time_diff:6.3f}s"
        )

train_state.params["log_std"] = -20.0 * jnp.ones((1,))

states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

state = env.prior_dist.sample(seed=state_key)
obs = env.obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(env.num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, obs, carry, train_state.params)
    state = env.trans_model.sample(state_key, state, action[0])
    obs = env.obs_model.sample(obs_key, state)

    states.append(state)
    actions.append(action[0])
    observations.append(obs)

# Convert lists to arrays for plotting
states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))

# Plot the results
plt.figure()
plt.plot(states[:, 0], states[:, 2], label="Trajectory")
plt.plot([-200], [100], "o", color="black", markersize=10, label="Starting point")
plt.plot([0], [0], "o", color="orange", markersize=10, label="Target")
plt.plot([-200, 0], [100, 0], "r--")
plt.title("Simulated trajectory")
plt.xlabel("x")
plt.ylabel("y")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
