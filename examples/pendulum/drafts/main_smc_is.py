import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

import optax

from jax import Array, random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from brax.training.replay_buffers import UniformSamplingQueue
from distrax import Block

from ppomdp.smc import smc, backward_tracing
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_pathwise_weighted,
)
from ppomdp.utils import policy_evaluation

import time
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import NamedTuple

from ppomdp.envs.pomdps import PendulumEnv as env


# Define a structure for buffer data
class TrajectorySample(NamedTuple):
    actions: Array
    observations: Array
    log_probs: Array
    log_marginal: Array


rng_key = random.PRNGKey(2)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.001
tempering = 0.75

learning_rate = 3e-4
batch_size = 16
num_epochs = 100

max_buffer_size = 512
min_buffer_size = 256
num_batches_per_epoch = 32

encoder = GRUEncoder(
    feature_fn=lambda x: x,
    dense_sizes=(256, 256),
    recurr_sizes=(128, 128),
    use_layer_norm=True,
)
decoder = NeuralGaussDecoder(
    decoder_sizes=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(1.0)),
)
bijector = Block(Tanh(), ndims=1)
policy = create_recurrent_neural_gauss_policy(
    encoder=encoder,
    decoder=decoder,
    bijector=bijector
)
key, sub_key = random.split(rng_key, 2)
params = policy.init(
    rng_key=sub_key,
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    batch_dim=num_history_particles,
)
learner = TrainState.create(
    params=params,
    apply_fn=lambda *_: None,
    tx=optax.rmsprop(learning_rate)
)

# Initialize the replay buffer
dummy_sample = TrajectorySample(
    actions=jnp.zeros((env.num_time_steps + 1, policy.dim)),
    observations=jnp.zeros((env.num_time_steps + 1, env.obs_dim)),
    log_probs=jnp.zeros((1,)),
    log_marginal=jnp.zeros((1,))
)
buffer_obj = UniformSamplingQueue(
    max_replay_size=max_buffer_size,
    dummy_data_sample=dummy_sample,
    sample_batch_size=batch_size
)
key, sub_key = random.split(key)
buffer_state = buffer_obj.init(sub_key)

num_steps = 0

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    avg_reward, *_ = policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy,
        params=learner.params,
        num_samples=1024
    )

    # run nested smc
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = \
        smc(
            rng_key=sub_key,
            num_time_steps=env.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            belief_prior=env.prior_dist,
            policy_prior=policy,
            policy_prior_params=learner.params,
            trans_model=env.trans_model,
            obs_model=env.obs_model,
            reward_fn=env.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering
        )

    num_steps += (env.num_time_steps + 1) * num_history_particles

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = \
        backward_tracing(sub_key, history_states, belief_states, belief_infos)

    # Calculate pathwise log probabilities for the generated trajectories
    log_probs = policy.pathwise_log_prob(
        actions=traced_history.actions,
        observations=traced_history.observations,
        params=learner.params
    )

    # Insert new data into the buffer
    samples = TrajectorySample(
        actions=jnp.swapaxes(traced_history.actions, 0, 1),
        observations=jnp.swapaxes(traced_history.observations, 0, 1),
        log_probs=log_probs,
        log_marginal=log_marginal * jnp.ones_like(log_probs)
    )
    buffer_state = buffer_obj.insert(buffer_state, samples)

    target_params = deepcopy(learner.params)
    target_log_marginal = log_marginal

    # Train the policy using samples from the buffer
    if buffer_obj.size(buffer_state) >= min_buffer_size:
        for _ in range(num_batches_per_epoch):

            # Sample a batch from the buffer
            key, sub_key = random.split(key)
            buffer_state, buffer_batch = buffer_obj.sample(buffer_state)

            sample_actions = jnp.swapaxes(buffer_batch.actions, 0, 1)
            sample_observations = jnp.swapaxes(buffer_batch.observations, 0, 1)
            sample_log_probs = jnp.squeeze(buffer_batch.log_probs)
            sample_log_marginal = jnp.squeeze(buffer_batch.log_marginal)

            target_log_probs = policy.pathwise_log_prob(
                actions=sample_actions,
                observations=sample_observations,
                params=target_params
            )

            log_weights = (target_log_probs - sample_log_probs) + (sample_log_marginal - target_log_marginal)
            importance_weights = jnp.exp(log_weights)

            # Perform training step with pathwise importance sampling
            learner, _ = train_recurrent_neural_gauss_policy_pathwise_weighted(
                learner=learner,
                policy=policy,
                actions=sample_actions,
                observations=sample_observations,
                importance_weights=importance_weights,
            )

    entropy = policy.entropy(learner.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal / tempering:.3f}, "
        f"Reward: {avg_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

key, sub_key = random.split(key)
_, states, actions = policy_evaluation(
    rng_key=sub_key,
    env_obj=env,
    policy=policy,
    params=learner.params,
    num_samples=16
)

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle("Simulated trajectories")

axs[0].plot(states[..., 0])
axs[0].set_ylabel("Angle")
axs[0].grid(True)

axs[1].plot(states[..., 1])
axs[1].set_ylabel("Angular Velocity")
axs[1].grid(True)

axs[2].plot(actions[..., 0])
axs[2].set_ylabel("Actions")
axs[2].grid(True)

plt.tight_layout()
plt.show()