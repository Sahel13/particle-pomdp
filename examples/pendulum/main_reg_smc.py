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

from ppomdp.smc import regularized_smc, backward_tracing

from ppomdp.bijector import Tanh
from ppomdp.utils import (
    batch_data,
    damping_schedule,
    policy_evaluation,
    flatten_particle_trajectories
)
from ppomdp.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss_alt import (
    create_recurrent_neural_gauss_policy,
    train_multihead_recurrent_neural_gauss_policy_stepwise,
)

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import PendulumEnv as env


rng_key = random.PRNGKey(0)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.005
tempering = 0.5
init_damping = 0.05
max_damping = 0.95

learning_rate = 1e-4
batch_size = 64
num_epochs = 100


policy_joint_bijector = Block(Tanh(), ndims=1)

policy_joint_encoder = GRUEncoder(
    feature_fn=lambda x: x,
    encoder_size=(256, 256),
    recurr_size=(64, 64),
)

###
policy_prior_decoder = NeuralGaussDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(2.0)),
)
key, sub_key = random.split(rng_key, 2)
policy_prior = create_recurrent_neural_gauss_policy(
    encoder=policy_joint_encoder,
    decoder=policy_prior_decoder,
    bijector=policy_joint_bijector
)
policy_prior_params = policy_prior.init(
    rng_key=sub_key,
    obs_dim=env.obs_dim,
    batch_dim=num_history_particles,
)

###
policy_posterior_decoder = NeuralGaussDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(2.0)),
)
key, sub_key = random.split(key, 2)
policy_posterior = create_recurrent_neural_gauss_policy(
    encoder=policy_joint_encoder,
    decoder=policy_posterior_decoder,
    bijector=policy_joint_bijector
)
policy_posterior_params = policy_posterior.init(
    rng_key=sub_key,
    obs_dim=env.obs_dim,
    batch_dim=num_history_particles,
)

joint_params = {
    "encoder": policy_prior_params["encoder"],
    "prior_decoder": policy_prior_params["decoder"],
    "posterior_decoder": policy_posterior_params["decoder"]
}
learner = TrainState.create(
    params=joint_params,
    apply_fn=lambda *_: None,
    tx=optax.adam(learning_rate)
)



num_steps = 0

# The training loop
for i in range(1, num_epochs + 1):
    start_time = time.time()

    policy_prior_params = {
        "encoder": learner.params["encoder"],
        "decoder": learner.params["prior_decoder"]
    }
    policy_posterior_params = {
        "encoder": learner.params["encoder"],
        "decoder": learner.params["posterior_decoder"]
    }

    # update damping param
    damping = damping_schedule(
        step=i,
        total_steps=num_epochs,
        steepness=1.,
        init_value=init_damping,
        max_value=max_damping
    )

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    expected_reward, *_ = policy_evaluation(
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
        regularized_smc(
            rng_key=sub_key,
            num_time_steps=env.num_time_steps,
            num_history_particles=num_history_particles,
            num_belief_particles=num_belief_particles,
            init_prior=env.prior_dist,
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
    actions, carry, observations = flatten_particle_trajectories(traced_history)
    data_size, _ = observations.shape

    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)
    for batch_idx in batch_indices:
        action_batch = jax.tree.map(lambda x: x[batch_idx, ...], actions)
        carry_batch = jax.tree.map(lambda x: x[batch_idx, ...], carry)
        observations_batch = jax.tree.map(lambda x: x[batch_idx, ...], observations)

        learner, _ = train_multihead_recurrent_neural_gauss_policy_stepwise(
            policy_prior=policy_prior,
            policy_posterior=policy_posterior,
            learner=learner,
            actions=action_batch,
            carry=carry_batch,
            observations=observations_batch,
            damping=damping
        )

    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal:.3f}, "
        f"Reward: {expected_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Damping: {damping:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )


policy_posterior_params = {
    "encoder": learner.params["encoder"],
    "decoder": learner.params["posterior_decoder"]
}

key, sub_key = random.split(key)
expected_reward, states, actions = \
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
