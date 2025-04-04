import csv
import os
from functools import partial

import common
import jax
from distrax import Block
from flax.linen.initializers import constant
from jax import numpy as jnp
from jax import random

from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.bijector import Tanh
from ppomdp.core import HistoryParticles
from ppomdp.gauss import (
    RecurrentNeuralGauss,
    create_recurrent_gauss_policy,
)
from ppomdp.smc import backward_tracing, smc
from ppomdp.utils import batch_data, custom_split


@jax.jit
def prepare_training_data(particles: HistoryParticles) -> HistoryParticles:
    if particles.observations.ndim != 3:
        raise ValueError("`particles` must be the output from smoothing.")

    actions = particles.actions[1:]
    particles = jax.tree.map(lambda x: x[:-1], particles)
    particles = particles._replace(actions=actions)
    return jax.tree.map(lambda x: x.reshape((-1, x.shape[-1])), particles)


@partial(jax.jit, static_argnames="policy")
def train_recurrent_gauss_policy(policy, train_state, particles, batch_idx):
    if particles.observations.ndim != 2:
        raise ValueError(
            "`particles.observations` must have shape `(batch_dim, obs_dim)`."
        )

    data = jax.tree.map(lambda x: x[batch_idx], particles)

    def loss_fn(params):
        log_probs = policy.log_prob(data.actions, data.carry, data.observations, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


@partial(jax.jit, static_argnames=("env", "policy", "num_samples"))
def evaluate(key, env, policy, train_state, num_samples=100):
    """Deploy the (deterministic) policy to sample trajectories and evaluate the average reward."""
    eval_params = train_state.params.copy()
    eval_params["log_std"] = -20.0 * jnp.ones_like(eval_params["log_std"])

    def body(carry, _key):
        states, policy_carry, observations, t = carry

        # Sample actions.
        _key, action_key = random.split(_key)
        policy_carry, actions = policy.sample(
            action_key, policy_carry, observations, eval_params
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


# Configuration
cmd_args = common.get_cmd_args()
env = common.get_env(cmd_args.env)
key = random.key(cmd_args.seed)

total_timesteps = int(1e6)
num_history_particles = 128
num_belief_particles = 64
tempering = 0.5
learning_rate = 1e-4
batch_size = 256
slew_rate_penalty = 5e-2

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
    init_log_std=constant(jnp.log(2.0)),
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


# The training loop.
while num_steps <= total_timesteps:
    # Run the particle filter.
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = smc(
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

    # Get the smoothed trajectories using genealogy tracking.
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = backward_tracing(
        sub_key, history_states, belief_states, belief_infos
    )

    # Update policy parameters.
    training_data = prepare_training_data(traced_history)
    num_data_points = training_data.observations.shape[0]
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_data_points, batch_size)
    for batch_idx in batch_indices:
        train_state, _ = train_recurrent_gauss_policy(
            policy, train_state, training_data, batch_idx
        )

    # Evaluate the policy.
    key, sub_key = random.split(key)
    expected_reward, states, actions = evaluate(sub_key, env, policy, train_state)
    logger.append([num_steps, expected_reward])

    print(
        f"Step: {num_steps:7d} | "
        f"Log marginal: {log_marginal:8.3f} | "
        f"Expected reward: {expected_reward:8.3f} | "
        f"Log std: {train_state.params['log_std'][0]:8.4f}"
    )

# Save the logging data.
with open(file_path, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(logger)
