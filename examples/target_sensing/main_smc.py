import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
jax.config.update("jax_enable_x64", True)

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling

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

from ppomdp.envs.pomdps import TargetEnv as env


rng_key = random.PRNGKey(1337)

num_history_particles = 512
num_belief_particles = 256

slew_rate_penalty = 0.0
tempering = 0.1

learning_rate = 1e-3
batch_size = 64
num_epochs = 500

encoder = GRUEncoder(
    feature_fn=env.feature_fn,
    encoder_size=(256, 256),
    recurr_size=(32, 32),
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
    batch_dim=num_history_particles,
    learning_rate=learning_rate
)

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current policy
    key, sub_key = random.split(key)
    history_states, _, _, _ = \
        smc(
            sub_key,
            env.num_time_steps,
            int(4 * num_history_particles),
            int(4 * num_belief_particles),
            env.prior_dist,
            env.trans_model,
            env.obs_model,
            policy,
            train_state.params,
            env.reward_fn,
            tempering=0.0,
            slew_rate_penalty=0.0,
        )
    expected_reward = jnp.mean(jnp.sum(history_states.rewards, axis=0))


    # run nested conditional smc
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = \
        smc(
            sub_key,
            env.num_time_steps,
            num_history_particles,
            num_belief_particles,
            env.prior_dist,
            env.trans_model,
            env.obs_model,
            policy,
            train_state.params,
            env.reward_fn,
            tempering,
            slew_rate_penalty,
        )

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = \
        backward_tracing(sub_key, history_states, belief_states, belief_infos)

    # # backward sample history states
    # key, sub_key = random.split(key)
    # traced_history, traced_belief = mcmc_backward_sampling(
    #     sub_key,
    #     num_history_particles,
    #     history_states,
    #     belief_states,
    #     env.trans_model,
    #     policy,
    #     train_state.params,
    #     env.reward_fn,
    #     tempering,
    #     slew_rate_penalty
    # )

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_history_particles, batch_size)
    for batch_idx in batch_indices:
        history_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history)
        train_state, batch_loss = \
            train_recurrent_gauss_policy(policy, train_state, history_batch)
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

state = env.prior_dist.sample(seed=state_key)
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
plt.figure()
plt.title("Simulated trajectory")
plt.plot(states[:, 0], states[:, 2], label="Trajectory")
plt.plot([-200], [100], "o", color="black", markersize=10, label="Starting point")
plt.plot([0], [0], "o", color="orange", markersize=10, label="Target")
plt.plot([-200, 0], [100, 0], "r--")
plt.xlabel("x")
plt.ylabel("y")
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
