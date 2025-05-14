import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
# jax.config.update("jax_enable_x64", True)

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
    train_attention_policy
)
from ppomdp.utils import batch_data, prepare_trajectories, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import BallCatchingEnv as env
from ppomdp.smc.utils import weighted_covar


rng_key = random.PRNGKey(1337)

num_history_particles = 128
num_belief_particles = 128
num_target_samples = 256

slew_rate_penalty = 0.0
tempering = 0.25

learning_rate = 3e-4
batch_size = 256
num_epochs = 500

bijector = Block(Tanh(), ndims=1)
encoder = AttentionEncoder(
    feature_fn=lambda x: x,
    hidden_size=128,
    attention_size=128,
    output_dim=128,
    num_heads=8,
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

# history_states, belief_states, belief_infos, log_marginal = \
#     smc(
#         rng_key=sub_key,
#         num_time_steps=env.num_time_steps,
#         num_history_particles=num_history_particles,
#         num_belief_particles=num_belief_particles,
#         belief_prior=env.belief_prior,
#         policy_prior=policy,
#         policy_prior_params=learner.params,
#         trans_model=env.trans_model,
#         obs_model=env.obs_model,
#         reward_fn=env.reward_fn,
#         slew_rate_penalty=slew_rate_penalty,
#         tempering=0.0,
#         history_resample_fn=systematic_resampling,
#         belief_resample_fn=systematic_resampling,
#     )
#
# key, sub_key = random.split(key)
# traced_history, traced_belief, traced_info = backward_tracing(
#     rng_key=sub_key,
#     history_states=history_states,
#     belief_states=belief_states,
#     belief_infos=belief_infos
# )
#
# key, sub_key = random.split(key)
# trajs = traced_info.mean
# ids = random.randint(key, (1,), 0, num_history_particles)
#
# plt.figure()
# plt.title("Simulated trajectory")
#
# # Plot trajectories
# plt.plot(trajs[:, ids, 0], trajs[:, ids, 1], alpha=0.7, label="Ball")
# plt.plot(trajs[:, ids, 3], trajs[:, ids, 4], alpha=0.7, label="Catcher")
#
# # Configure plot
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# # plt.gca().set_aspect("equal", adjustable="box")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# The training loop
while num_steps <= int(2e6):
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

    num_steps += env.num_time_steps * num_history_particles

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

    print(
        f"Num steps: {num_steps:6d}, "
        f"Log marginal: {log_marginal / tempering:.3f}, "
        f"Reward: {avg_return:.3f}, "
        f"Entropy: {entropy:.3f}, "
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

covars = weighted_covar(beliefs.particles[:, 0], beliefs.weights[:, 0])
covars = jnp.diagonal(covars, axis1=-2, axis2=-1)

plt.figure()
plt.title("Simulated trajectory")

# Plot trajectories
plt.plot(states[:, 0, 0], states[:, 0, 1], label="Ball")
plt.plot(states[:, 0, 6], states[:, 0, 7], label="Catcher")

# Configure plot
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.title("Covariance of belief particles")
plt.plot(2 * jnp.sqrt(covars[:, 0]), label="2 std")
plt.show()


# plt.figure()
# plt.show()
#
plt.figure()
plt.plot(states[:, 0, -2] * 180 / jnp.pi, label="phi")
plt.plot(actions[:, 0, -1] * 180 / jnp.pi, label="theta")
plt.legend()
plt.show()
