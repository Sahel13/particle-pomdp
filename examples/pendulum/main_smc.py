import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_stepwise,
)
from ppomdp.utils import (
    batch_data,
    policy_evaluation,
    flatten_particle_trajectories
)

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import PendulumEnv as env


rng_key = random.PRNGKey(1)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.0
tempering = 0.5

learning_rate = 3e-4
batch_size = 256
num_epochs = 100

encoder = GRUEncoder(
    feature_fn=lambda x: x,
    dense_sizes=(256, 128),
    recurr_sizes=(64, 64),
    use_layer_norm=True,

)
decoder = NeuralGaussDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(2.0)),
)
bijector = Block(Tanh(), ndims=1)

key, sub_key = random.split(rng_key, 2)
policy = create_recurrent_neural_gauss_policy(
    encoder=encoder,
    decoder=decoder,
    bijector=bijector
)
params = policy.init(
    rng_key=sub_key,
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    batch_dim=num_history_particles,
)
learner = TrainState.create(
    params=params,
    apply_fn=lambda *_: None,
    tx=optax.adam(learning_rate)
)


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
            init_prior=env.prior_dist,
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

    # update policy parameters
    actions, next_actions, observations, next_observations, carry = \
        flatten_particle_trajectories(traced_history)
    data_size, _ = observations.shape

    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)
    for batch_idx in batch_indices:
        action_batch = jax.tree.map(lambda x: x[batch_idx, ...], actions)
        next_action_batch = jax.tree.map(lambda x: x[batch_idx, ...], next_actions)
        observations_batch = jax.tree.map(lambda x: x[batch_idx, ...], observations)
        carry_batch = jax.tree.map(lambda x: x[batch_idx, ...], carry)

        learner, batch_loss = train_recurrent_neural_gauss_policy_stepwise(
            policy=policy,
            learner=learner,
            next_actions=next_action_batch,
            carry=carry_batch,
            actions=action_batch,
            observations=observations_batch,
        )
        loss += batch_loss

    entropy = policy.entropy(learner.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal:.3f}, "
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
