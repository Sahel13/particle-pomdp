import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

jax.config.update("jax_enable_x64", True)

from ppomdp.core import Reference
from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling
from ppomdp.csmc import csmc

from ppomdp.bijector import Tanh
from ppomdp.utils import batch_data
from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.gauss import (
    RecurrentNeuralGauss,
    create_recurrent_gauss_policy,
    train_recurrent_gauss_policy
)

import time
from copy import deepcopy
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import PendulumEnv as env


rng_key = random.PRNGKey(10)

num_outer_particles = 256
num_inner_particles = 256

slew_rate_penalty = 0.001
tempering = 0.5
num_moves = 1

learning_rate = 1e-3
batch_size = 32
num_epochs = 500

encoder = GRUEncoder(
    feature_fn=lambda x: x,
    encoder_size=(256, 256),
    recurr_size=(64, 64),
)

decoder = MLPDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
)

network = RecurrentNeuralGauss(
    encoder=encoder,
    decoder=decoder,
    init_log_std=constant(jnp.log(1.0)),
)

bijector = Block(Tanh(), ndims=1)

key, sub_key = random.split(rng_key, 2)
policy = create_recurrent_gauss_policy(network, bijector)
train_state = policy.init(
    rng_key=sub_key,
    input_dim=env.obs_dim,
    output_dim=env.action_dim,
    batch_dim=num_outer_particles,
    learning_rate=learning_rate
)

# run init nested smc
key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, _ = \
    smc(
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
        slew_rate_penalty
    )

# trace ancestors of outer states
key, sub_key = random.split(key)
traced_outer, traced_inner, _ = \
    backward_tracing(sub_key, outer_states, inner_states, inner_infos)

# # backward sample outer states
# key, sub_key = random.split(key)
# traced_outer, traced_inner = mcmc_backward_sampling(
#     sub_key,
#     num_outer_particles,
#     outer_states,
#     inner_states,
#     env.trans_model,
#     policy,
#     train_state.params,
#     env.reward_fn,
#     tempering,
#     slew_rate_penalty
# )

# sample a new reference
key, sub_key = random.split(key)
idx = jax.random.choice(sub_key, jnp.arange(num_outer_particles))
reference = Reference(
    outer_particles=jax.tree.map(lambda x: x[:, idx], traced_outer),
    inner_state=jax.tree.map(lambda x: x[:, idx], traced_inner)
)

for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current policy
    key, sub_key = random.split(key)
    outer_states, _, _, _ = \
        smc(
            sub_key,
            env.num_time_steps,
            int(4 * num_outer_particles),
            int(4 * num_inner_particles),
            env.prior_dist,
            env.trans_model,
            env.obs_model,
            policy,
            train_state.params,
            env.reward_fn,
            tempering=0.0,
            slew_rate_penalty=0.0,
        )
    expected_reward = jnp.mean(jnp.sum(outer_states.rewards, axis=0))

    for _ in range(num_moves):
        # run nested conditional smc
        key, sub_key = random.split(key)
        outer_states, inner_states, inner_info, log_marginal = \
            csmc(
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
                reference
            )

        # trace ancestors of outer states
        key, sub_key = random.split(key)
        traced_outer, traced_inner, _ = \
            backward_tracing(sub_key, outer_states, inner_states, inner_info)

        # # backward sample outer states
        # key, sub_key = random.split(key)
        # traced_outer, traced_inner = mcmc_backward_sampling(
        #     sub_key,
        #     num_outer_particles,
        #     outer_states,
        #     inner_states,
        #     env.trans_model,
        #     policy,
        #     train_state.params,
        #     env.reward_fn,
        #     tempering,
        #     slew_rate_penalty
        # )

        # sample a new reference
        key, sub_key = random.split(key)
        idx = jax.random.choice(sub_key, jnp.arange(num_outer_particles))
        reference = Reference(
            outer_particles=jax.tree.map(lambda x: x[:, idx], traced_outer),
            inner_state=jax.tree.map(lambda x: x[:, idx], traced_inner)
        )

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_outer_particles, batch_size)
    for batch_idx in batch_indices:
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer)
        train_state, batch_loss = \
            train_recurrent_gauss_policy(policy, train_state, outer_batch)
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
eval_state.params["log_std"] = -20.0 * jnp.ones((env.action_dim,))

# plot realization
states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

state = jnp.array([0.0, 0.0])
obs = env.obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(env.num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, carry, obs, train_state.params)
    state = env.trans_model.sample(state_key, state, action[0])
    obs = env.obs_model.sample(obs_key, state)

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
