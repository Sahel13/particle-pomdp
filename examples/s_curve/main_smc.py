import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import time
from functools import partial

import jax
from jax import random
import jax.numpy as jnp

from ppomdp.smc import backward_tracing, smc
from ppomdp.bijector import Tanh
from ppomdp.policy import LSTM, create_policy, train_policy
from ppomdp.utils import batch_data, weighted_mean

from distrax import Chain, ScalarAffine
from flax.linen.initializers import constant
from flax.training.train_state import TrainState

import optax
from copy import deepcopy
import matplotlib.pyplot as plt

from environment import prior_dist, trans_model, obs_model, reward_fn
from environment import state_dim, action_dim, obs_dim, num_time_steps

jax.config.update("jax_enable_x64", True)


@partial(jnp.vectorize, signature="(m)->(n)")
def feature_fn(z: jax.Array) -> jax.Array:
    return jnp.array((jnp.sin(z[0]), jnp.cos(z[0])))


network = LSTM(
    dim=action_dim,
    feature_fn=feature_fn,
    encoder_size=[256, 256],
    recurr_size=[32, 32],
    output_size=[256, 256],
    init_log_std=constant(jnp.log(1.0)),
)
bijector = Chain([ScalarAffine(0.0, 9.0 / 180 * jnp.pi), Tanh()])
policy = create_policy(network, bijector)

rng_key = random.PRNGKey(1)

num_outer_particles = 512
num_inner_particles = 256
tempering = 0.1
slew_rate_penalty = 0.0

learning_rate = 1e-3
batch_size = 32
num_epochs = 500

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

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current policy
    eval_state = deepcopy(train_state)
    eval_state.params["log_std"] = -20.0 * jnp.ones((action_dim,))

    key, sub_key = random.split(key)
    outer_states, _, _, _ = \
        jitted_smc(
            sub_key,
            num_time_steps,
            int(4 * num_outer_particles),
            int(4 * num_inner_particles),
            prior_dist,
            trans_model,
            obs_model,
            policy,
            eval_state.params,
            reward_fn,
            tempering=0.0,
            slew_rate_penalty=0.0,
        )
    expected_reward = jnp.mean(jnp.sum(outer_states.rewards, axis=0))

    # run nested smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_infos, log_marginal = \
        jitted_smc(
            sub_key,
            num_time_steps,
            num_outer_particles,
            num_inner_particles,
            prior_dist,
            trans_model,
            obs_model,
            policy,
            train_state.params,
            reward_fn,
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
        train_state, batch_loss = train_policy(policy, train_state, outer_batch)
        loss += batch_loss

    entropy = policy.entropy(train_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Reward: {expected_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

eval_state = deepcopy(train_state)
eval_state.params["log_std"] = -20.0 * jnp.ones((action_dim,))

# plot realization
states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

state = prior_dist.sample(seed=state_key)
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
