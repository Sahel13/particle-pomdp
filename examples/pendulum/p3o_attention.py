import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

import optax

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.smc import smc, backward_tracing
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import AttentionEncoder, NeuralGaussDecoder
from ppomdp.policy.attention import (
    create_attention_policy,
    train_attention_policy,
)
from ppomdp.utils import batch_data, prepare_trajectories, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import PendulumEnv as env


rng_key = random.PRNGKey(1)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.05
tempering = 0.5

learning_rate = 3e-4
batch_size = 256
num_epochs = 100

bijector = Block(Tanh(), ndims=1)
encoder = AttentionEncoder(
    feature_fn=env.feature_fn,
    hidden_size=128,
    attention_size=128,
    output_dim=128,
    num_heads=4,
)
decoder = NeuralGaussDecoder(
    decoder_sizes=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(1.0)),
)
policy = create_attention_policy(
    encoder=encoder,
    decoder=decoder,
    bijector=bijector
)

key, sub_key = random.split(rng_key, 2)
params = policy.init(
    rng_key=sub_key,
    particle_dim=env.state_dim,
    action_dim=env.action_dim,
    batch_size=num_history_particles,
    num_particles=num_belief_particles,
)
learner = TrainState.create(
    params=params,
    apply_fn=lambda *_: None,
    tx=optax.adam(learning_rate),
)

num_steps = 0

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    rewards, *_ = policy_evaluation(
        rng_key=sub_key,
        num_time_steps=env.num_time_steps,
        num_trajectory_samples=1024,
        num_belief_particles=num_belief_particles,
        init_dist=env.init_dist,
        belief_prior=env.belief_prior,
        policy=policy,
        policy_params=learner.params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        stochastic=False
    )
    avg_return = jnp.mean(jnp.sum(rewards, axis=0))

    # run nested smc
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = \
        smc(
            rng_key=sub_key,
            num_time_steps=env.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            belief_prior=env.belief_prior,
            policy_prior=policy,
            policy_prior_params=learner.params,
            trans_model=env.trans_model,
            obs_model=env.obs_model,
            reward_fn=env.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
            history_resample_fn=systematic_resampling,
            belief_resample_fn=systematic_resampling,
        )

    num_steps += (env.num_time_steps + 1) * num_history_particles

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = backward_tracing(
        rng_key=sub_key,
        history_states=history_states,
        belief_states=belief_states,
        belief_infos=belief_infos
    )

    # update policy parameters
    key, sub_key = random.split(key)
    actions, particles, weights = \
        prepare_trajectories(sub_key, traced_history.actions, traced_belief)

    data_size, _ = actions.shape
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)

    loss = 0.0
    for batch_idx in batch_indices:
        action_batch = jax.tree.map(lambda x: x[batch_idx, ...], actions)
        particles_batch = jax.tree.map(lambda x: x[batch_idx, ...], particles)
        weights_batch = jax.tree.map(lambda x: x[batch_idx, ...], weights)

        learner, batch_loss = train_attention_policy(
            policy=policy,
            learner=learner,
            actions=action_batch,
            particles=particles_batch,
            weights=weights_batch,
        )
        loss += batch_loss

    entropy = policy.entropy(learner.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal / tempering:.3f}, "
        f"Reward: {avg_return:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )


key, sub_key = random.split(key)
_, states, actions, beliefs = policy_evaluation(
    rng_key=sub_key,
    num_time_steps=env.num_time_steps,
    num_trajectory_samples=1024,
    num_belief_particles=num_belief_particles,
    init_dist=env.init_dist,
    belief_prior=env.belief_prior,
    policy=policy,
    policy_params=learner.params,
    trans_model=env.trans_model,
    obs_model=env.obs_model,
    reward_fn=env.reward_fn,
    stochastic=False
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