from functools import partial
import os
import csv

import common
import jax
from distrax import Block
from flax.linen.initializers import constant
from jax import numpy as jnp
from jax import random

from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.bijector import Tanh
from ppomdp.core import Reference
from ppomdp.csmc import csmc
from ppomdp.gauss import (
    RecurrentNeuralGauss,
    create_recurrent_gauss_policy,
    train_recurrent_gauss_policy,
)
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data, custom_split, weighted_mean
import matplotlib.pyplot as plt


@partial(jax.jit, static_argnames=("env", "policy", "num_samples"))
def evaluate(key, env, policy, train_state, num_samples=100):
    """Deploy the policy to sample trajectories and evaluate the average reward."""

    def body(carry, _key):
        states, policy_carry, observations, t = carry

        # Sample actions.
        _key, action_key = random.split(_key)
        policy_carry, actions = policy.sample(
            action_key, policy_carry, observations, train_state.params
        )
        # Sample next states.
        _key, state_keys = custom_split(_key, num_samples + 1)
        states = jax.vmap(env.trans_model.sample)(state_keys, states, actions)
        # Compute rewards.
        rewards = jax.vmap(env.reward_fn, (0, 0, None))(states, actions, t)
        # Sample observations.
        obs_keys = random.split(_key, num_samples)
        observations = jax.vmap(env.obs_model.sample)(obs_keys, states)

        return (states, policy_carry, observations, t + 1), (states, actions, rewards)

    key, state_key = random.split(key)
    init_states = env.prior_dist.sample(seed=state_key, sample_shape=num_samples)
    key, obs_keys = custom_split(key, num_samples + 1)
    init_observations = jax.vmap(env.obs_model.sample)(obs_keys, init_states)
    init_policy_carry = policy.reset(num_samples)

    _, (states, actions, rewards) = jax.lax.scan(
        body,
        (init_states, init_policy_carry, init_observations, 1),
        random.split(key, env.num_time_steps),
    )
    states = jnp.concatenate([init_states[None], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))
    return expected_reward, states, actions


cmd_args = common.get_cmd_args()
env = common.get_env(cmd_args.env)
key = random.key(cmd_args.seed)

num_history_particles = 256
num_belief_particles = 256

slew_rate_penalty = 0.0
tempering = 0.5

learning_rate = 1e-3
batch_size = 64
num_epochs = 500

if cmd_args.env == "cartpole":
    tempering = 0.25
    batch_size = 32
    slew_rate_penalty = 5e-4
    num_epochs = 200

encoder = GRUEncoder(
    feature_fn=lambda x: x,
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

key, sub_key = random.split(key)
policy = create_recurrent_gauss_policy(network, bijector)
train_state = policy.init(
    rng_key=sub_key,
    input_dim=env.obs_dim,
    output_dim=env.action_dim,
    batch_dim=num_history_particles,
    learning_rate=learning_rate,
)

# Set up logging.
file_name = f"training_log_seed_{cmd_args.seed}.csv"
file_path = os.path.join(cmd_args.log_dir, file_name)
logger = [["Step", "Episodic reward"]]
num_steps = 0

# Check policy performance before training.
key, sub_key = random.split(key)
expected_reward, *_ = evaluate(sub_key, env, policy, train_state)
print(f"Step: {num_steps:7d} | Expected reward: {expected_reward:8.3f}")
logger.append([num_steps, expected_reward])

# run init nested smc
key, sub_key = random.split(key)
history_states, belief_states, belief_infos, _ = smc(
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
num_steps += num_history_particles * (env.num_time_steps + 1)

# trace ancestors of history states
key, sub_key = random.split(key)
traced_history, traced_belief, _ = backward_tracing(
    sub_key, history_states, belief_states, belief_infos
)

# mean_particles = weighted_mean(
#     traced_belief.particles, traced_belief.weights
# )
# actions = traced_history.actions[:, :, 0]
#
# fig, axs = plt.subplots(5, 1, figsize=(10, 15))
# fig.suptitle("Smoothed trajectories")
# axs[0].plot(mean_particles[:, :, 0])
# axs[0].set_ylabel("Cart position")
# axs[0].grid(True)
# axs[1].plot(mean_particles[:, :, 1])
# axs[1].set_ylabel("Pole angle")
# axs[1].grid(True)
# axs[2].plot(mean_particles[:, :, 2])
# axs[2].set_ylabel("Cart velocity")
# axs[2].grid(True)
# axs[3].plot(mean_particles[:, :, 3])
# axs[3].set_ylabel("Pole angular velocity")
# axs[3].grid(True)
# axs[4].plot(actions)
# axs[4].set_ylabel("Action")
# axs[4].set_xlabel("Time step")
# axs[4].grid(True)
# plt.tight_layout()
# plt.show()

# sample a new reference
key, sub_key = random.split(key)
idx = jax.random.choice(sub_key, jnp.arange(num_history_particles))
reference = Reference(
    history_particles=jax.tree.map(lambda x: x[:, idx], traced_history),
    belief_state=jax.tree.map(lambda x: x[:, idx], traced_belief),
)

# The training loop
for i in range(1, num_epochs + 1):
    # run nested conditional smc
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = csmc(
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
        reference,
    )

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = backward_tracing(
        sub_key, history_states, belief_states, belief_infos
    )

    # sample a new reference
    key, sub_key = random.split(key)
    idx = jax.random.choice(sub_key, jnp.arange(num_history_particles))
    reference = Reference(
        history_particles=jax.tree.map(lambda x: x[:, idx], traced_history),
        belief_state=jax.tree.map(lambda x: x[:, idx], traced_belief),
    )

    # update policy parameters
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_history_particles, batch_size)
    for batch_idx in batch_indices:
        history_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history)
        train_state, _ = train_recurrent_gauss_policy(
            policy, train_state, history_batch
        )

    entropy = policy.entropy(train_state.params)

    # Evaluate the policy
    num_steps += num_history_particles * (env.num_time_steps + 1)
    key, sub_key = random.split(key)
    expected_reward, states, actions = evaluate(sub_key, env, policy, train_state)
    logger.append([num_steps, expected_reward])

    print(
        f"Epoch: {i:3d} | "
        f"Step: {num_steps:7d} | "
        f"Log marginal: {log_marginal:8.3f} | "
        f"Expected reward: {expected_reward:8.3f} | "
        f"Log std: {train_state.params["log_std"][0]:8.4f} | "
    )

common.plot_trajectory(cmd_args.env, states, actions)

# Save the logging data.
with open(file_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(logger)
