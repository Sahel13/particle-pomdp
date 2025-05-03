import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

import optax

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.smc import rsmc, backward_tracing
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import (
    GRUEncoder,
    NeuralGaussDecoder
)
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    initialize_multihead_recurrent_gauss_policy,
    train_multihead_recurrent_neural_gauss_policy_pathwise
)
from ppomdp.utils import (
    batch_data,
    damping_schedule,
    policy_evaluation,
)

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import PendulumEnv as env


rng_key = random.PRNGKey(33)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.001
tempering = 0.5
init_damping = 0.05
max_damping = 0.75

learning_rate = 3e-4
batch_size = 32
num_epochs = 100

joint_bijector = Block(Tanh(), ndims=1)
joint_encoder = GRUEncoder(
    feature_fn=lambda x: x,
    dense_sizes=(256, 256),
    recurr_sizes=(64, 64),
    use_layer_norm=True
)

###
prior_decoder = NeuralGaussDecoder(
    decoder_sizes=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(1.0)),
)
policy_prior = create_recurrent_neural_gauss_policy(
    encoder=joint_encoder,
    decoder=prior_decoder,
    bijector=joint_bijector
)

###
posterior_decoder = NeuralGaussDecoder(
    decoder_sizes=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(1.0)),
)
policy_posterior = create_recurrent_neural_gauss_policy(
    encoder=joint_encoder,
    decoder=posterior_decoder,
    bijector=joint_bijector
)

key, sub_key = random.split(rng_key, 2)
joint_params = initialize_multihead_recurrent_gauss_policy(
    rng_key=sub_key,
    obs_dim=env.obs_dim,
    action_dim=env.action_dim,
    batch_dim=num_history_particles,
    encoder=joint_encoder,
    prior_decoder=prior_decoder,
    posterior_decoder=posterior_decoder,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adam(learning_rate)
)
learner = TrainState.create(
    apply_fn=None,
    params=joint_params,
    tx=optimizer
)

num_steps = 0

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    # update damping param
    damping = damping_schedule(
        step=i,
        total_steps=num_epochs,
        init_value=init_damping,
        max_value=max_damping
    )

    policy_prior_params = {
        "encoder": learner.params["encoder"],
        "decoder": learner.params["prior_decoder"]
    }
    policy_posterior_params = {
        "encoder": learner.params["encoder"],
        "decoder": learner.params["posterior_decoder"]
    }

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    avg_reward, *_ = policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy_posterior,
        params=policy_posterior_params,
        num_samples=1024
    )
    entropy = policy_posterior.entropy(policy_posterior_params)

    # run nested smc
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = \
        rsmc(
            rng_key=sub_key,
            num_time_steps=env.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            belief_prior=env.prior_dist,
            policy_prior=policy_prior,
            policy_prior_params=policy_prior_params,
            policy_posterior=policy_posterior,
            policy_posterior_params=policy_posterior_params,
            trans_model=env.trans_model,
            obs_model=env.obs_model,
            reward_fn=env.reward_fn,
            slew_rate_penalty=slew_rate_penalty,
            tempering=tempering,
            damping=damping,
        )

    num_steps += (env.num_time_steps + 1) * num_history_particles

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = \
        backward_tracing(sub_key, history_states, belief_states, belief_infos)

    # update policy parameters
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_history_particles, batch_size)
    for batch_idx in batch_indices:
        action_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.actions)
        observation_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.observations)

        learner, _ = \
            train_multihead_recurrent_neural_gauss_policy_pathwise(
                learner=learner,
                policy_prior=policy_prior,
                policy_posterior=policy_posterior,
                actions=action_batch,
                observations=observation_batch,
                damping=damping,
            )

    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal / tempering:.3f}, "
        f"Reward: {avg_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Damping: {damping:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

policy_posterior_params = {
    "encoder": learner.params["encoder"],
    "decoder": learner.params["posterior_decoder"]
}

key, sub_key = random.split(key)
_, states, actions = \
    policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy_posterior,
        params=policy_posterior_params,
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
