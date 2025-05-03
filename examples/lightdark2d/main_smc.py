import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
jax.config.update("jax_enable_x64", True)

import optax

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_pathwise
)
from ppomdp.utils import batch_data, policy_evaluation, policy_evaluation_with_beliefs
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ppomdp.envs.pomdps import LightDark2DEnv as env
from ppomdp.envs.pomdps.lightdark2d import stddev_obs


rng_key = random.PRNGKey(1337)

num_history_particles = 256
num_belief_particles = 64
num_target_samples = 512

slew_rate_penalty = 0.01
tempering = 0.5

learning_rate = 3e-4
batch_size = 128
num_epochs = 500

bijector = Block(Tanh(), ndims=1)
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
        init_dist=env.init_dist,
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
            belief_resample_fn=multinomial_resampling,
        )

    num_steps += (env.num_time_steps + 1) * num_history_particles

    # # trace ancestors of history states
    # key, sub_key = random.split(key)
    # traced_history, traced_belief, _ = \
    #     backward_tracing(sub_key, history_states, belief_states, belief_infos)

    # backward sample history states
    key, sub_key = random.split(key)
    traced_history, traced_belief = mcmc_backward_sampling(
        rng_key=sub_key,
        num_samples=num_target_samples,
        policy_prior=policy,
        policy_prior_params=learner.params,
        trans_model=env.trans_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=slew_rate_penalty,
        tempering=tempering,
        history_states=history_states,
        belief_states=belief_states,
    )

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_target_samples, batch_size)
    for batch_idx in batch_indices:
        action_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.actions)
        observation_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.observations)

        learner, batch_loss = train_recurrent_neural_gauss_policy_pathwise(
            policy=policy,
            learner=learner,
            actions=action_batch,
            observations=observation_batch,
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
rewards, states, actions, beliefs = policy_evaluation_with_beliefs(
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
def plot_covariance_ellipse(ax, data, mean, color):
    """Calculates and plots a covariance ellipse for 2D data."""
    covar = jnp.cov(data, rowvar=False)
    eigvals, eigvecs = jnp.linalg.eigh(covar)
    angle = jnp.degrees(jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = jnp.sqrt(jnp.maximum(eigvals, 1e-9)) # Use std dev for ellipse size
    ell = patches.Ellipse(mean, width, height, angle=angle, edgecolor=color, facecolor='none')
    ax.add_patch(ell)

fig_env, ax_env = plt.subplots(1, 1, figsize=(8, 6))

# Plot environment background (observation noise level)
xgrid = jnp.linspace(-1.0, 6.0, 100)
ygrid = jnp.linspace(-2.0, 2.5, 100)
X, Y = jnp.meshgrid(xgrid, ygrid)

Z = jnp.zeros_like(X)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        # Assuming state includes position (x, y) and potentially velocity (vx, vy)
        # Construct a state vector at the grid point with zero velocity if needed
        state_at_grid = jnp.array([X[r, c], Y[r, c]] + [0.0] * (env.state_dim - 2))
        if env.state_dim >= 2:
            light_level = jnp.linalg.norm(stddev_obs(state_at_grid[:env.state_dim]))
            Z = Z.at[r, c].set(light_level)


im = ax_env.imshow(-Z, extent=(xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()), origin='lower', cmap='gray', aspect='auto')
plt.colorbar(im, ax=ax_env, label='Observation Noise Magnitude (Negative)')
ax_env.set_title('Light-Dark Environment with State and Belief Evolution')
ax_env.set_xlabel('X Position')
ax_env.set_ylabel('Y Position')

# Plot mean state and belief trajectories
mean_states_2d = jnp.mean(states[:, :, :2], axis=1)
mean_beliefs_2d = jnp.mean(beliefs.particles[:, :, :, :2], axis=(1, 2))

ax_env.plot(mean_states_2d[:, 0], mean_states_2d[:, 1], 'r-')
ax_env.plot(mean_beliefs_2d[:, 0], mean_beliefs_2d[:, 1], 'g-')

# Plot covariance ellipses at intervals
plot_ellipse_interval = max(1, env.num_time_steps // 10) # Adjust interval as needed
for t in range(0, env.num_time_steps + 1, plot_ellipse_interval):
    plot_covariance_ellipse(ax_env, states[t, :, :2], mean_states_2d[t], 'b')
    plot_covariance_ellipse(ax_env, jnp.mean(beliefs.particles[t, :, :, :2], axis=1), mean_beliefs_2d[t], 'm')

plt.show()