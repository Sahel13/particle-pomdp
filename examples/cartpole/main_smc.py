import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
jax.config.update("jax_enable_x64", True)

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling
from ppomdp.bijector import Tanh
from ppomdp.utils import batch_data, flatten_particle_trajectories, policy_evaluation
from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.gauss import (
    RecurrentNeuralGauss,
    create_recurrent_gauss_policy,
    train_recurrent_gauss_policy_stepwise,
    train_recurrent_gauss_policy_pathwise
)

import time
from copy import deepcopy
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import CartPoleEnv as env


rng_key = random.PRNGKey(123)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.05
tempering = 0.5

learning_rate = 1e-4
batch_size = 256
num_epochs = 200

encoder = GRUEncoder(
    feature_fn=lambda x: x,
    encoder_size=(256, 256),
    recurr_size=(128, 128),
)
decoder = MLPDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
)
network = RecurrentNeuralGauss(
    encoder=encoder,
    decoder=decoder,
    init_log_std=constant(1.0),
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

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    expected_reward, *_ = \
        policy_evaluation(sub_key, env, policy, train_state.params)

    # run nested conditional smc
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, _ = \
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

    # flatten traced history
    flat_traced_history = flatten_particle_trajectories(traced_history)
    data_size = flat_traced_history.observations.shape[0]

    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)
    for batch_idx in batch_indices:
        history_batch = jax.tree.map(lambda x: x[batch_idx, ...], flat_traced_history)
        train_state, batch_loss = train_recurrent_gauss_policy_stepwise(
            policy, train_state, history_batch
        )
        loss += batch_loss

    # loss = 0.0
    # key, sub_key = random.split(key)
    # batch_indices = batch_data(sub_key, num_history_particles, batch_size)
    # for batch_idx in batch_indices:
    #     history_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history)
    #     train_state, batch_loss = \
    #         train_recurrent_gauss_policy_pathwise(policy, train_state, history_batch)
    #     loss += batch_loss

    entropy = policy.entropy(train_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Reward: {expected_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

eval_params = deepcopy(train_state.params)
eval_params["log_std"] = -20.0 * jnp.ones((env.action_dim,))

states = []
actions = []
observations = []

key = random.PRNGKey(21)
key, state_key, obs_key = random.split(key, 3)

state = jnp.zeros(env.state_dim)
obs = env.obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(env.num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, carry, obs, eval_params)
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
axs[0].set_ylabel("Cart position")
axs[0].grid(True)

axs[1].plot(states[:, 1])
axs[1].set_ylabel("Pole angle")
axs[1].grid(True)

axs[2].plot(actions)
axs[2].set_ylabel("Action")
axs[2].grid(True)

plt.tight_layout()
plt.show()
