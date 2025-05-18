import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
jax.config.update("jax_enable_x64", True)

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
from ppomdp.smc.utils import systematic_resampling

import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ppomdp.envs.pomdps import LightDark2DEnv as env
from ppomdp.envs.pomdps.lightdark2d import stddev_obs


rng_key = random.PRNGKey(0)

num_history_particles = 256
num_belief_particles = 32
num_target_samples = 256

slew_rate_penalty = 0.05
tempering = 0.25

learning_rate = 3e-4
batch_size = 256
num_epochs = 500

bijector = Block(Tanh(), ndims=1)
encoder = AttentionEncoder(
    feature_fn=env.feature_fn,
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

# --- Plot 1: State and Action Trajectories ---
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
num_trajectories_to_plot = min(10, states.shape[1]) # Plot up to 10 trajectories
plot_indices = random.choice(key, states.shape[1], shape=(num_trajectories_to_plot,), replace=False)

axs[0].plot(states[:, plot_indices, 0])
axs[0].set_ylabel('State-1')

axs[1].plot(states[:, plot_indices, 1])
axs[1].set_ylabel('State-2')

axs[2].plot(actions[:, plot_indices, 0])
axs[2].set_ylabel('Act-1')

axs[3].plot(actions[:, plot_indices, 1])
axs[3].set_ylabel('Act-2')
axs[3].set_xlabel('Time Step')

plt.tight_layout()
plt.show()

# --- Plot 2: Environment, Mean Trajectories, and Covariance Ellipses ---

# Helper function to plot covariance ellipse
def plot_covar_ellipse(ax, mean, covar, color):
    """Calculates and plots a covariance ellipse for 2D data."""
    eigvals, eigvecs = jnp.linalg.eigh(covar)
    angle = jnp.degrees(jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = jnp.sqrt(jnp.maximum(eigvals, 1e-9)) # Use std dev for ellipse size
    ell = patches.Ellipse(mean, width, height, angle=angle, edgecolor=color, facecolor='none')
    ax.add_patch(ell)


# Plot environment background (observation noise level)
xgrid = jnp.linspace(-1.0, 6.0, 100)
ygrid = jnp.linspace(-2.0, 2.5, 100)
X, Y = jnp.meshgrid(xgrid, ygrid)

light_level = jnp.zeros_like(X)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        _state = jnp.array([X[r, c], Y[r, c]] + [0.0] * (env.state_dim - 2))
        _light_level = jnp.linalg.norm(stddev_obs(_state[:env.state_dim]))
        light_level = light_level.at[r, c].set(_light_level)


fig_env, ax_env = plt.subplots(1, 1, figsize=(8, 6))

im = ax_env.imshow(
    -light_level,
    extent=(xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()),
    origin='lower', cmap='gray', aspect='auto'
)
plt.colorbar(im, ax=ax_env, label='Observation Noise Magnitude (Negative)')
ax_env.set_title('Light-Dark Environment with State and Belief Evolution')
ax_env.set_xlabel('X Position')
ax_env.set_ylabel('Y Position')

# Plot mean state and belief trajectories
from ppomdp.smc.utils import weighted_mean, weighted_covar
belief_mean_per_traj = weighted_mean(beliefs.particles, beliefs.weights)
belief_covar_per_traj = weighted_covar(beliefs.particles, beliefs.weights)

belief_mean = jnp.mean(belief_mean_per_traj, axis=1)
belief_covar = jnp.mean(belief_covar_per_traj, axis=1)

state_mean = jnp.mean(states, axis=1)
state_center = states - state_mean[:, None, :]
state_covar = jnp.einsum('tnk,tnh->tkh', state_center, state_center) / state_center.shape[1]

# ax_env.plot(state_mean[:, 0], state_mean[:, 1], 'r-')
ax_env.plot(belief_mean[:, 0], belief_mean[:, 1], 'r-')

# Plot covariance ellipses at intervals
plot_ellipse_interval = max(1, env.num_time_steps // 10) # Adjust interval as needed
for t in range(0, env.num_time_steps + 1, plot_ellipse_interval):
    # plot_covar_ellipse(ax_env, state_mean[t, :2], state_covar[t, :2, :2], 'm')
    plot_covar_ellipse(ax_env, belief_mean[t, :2], belief_covar[t, :2, :2], 'g')

plt.show()