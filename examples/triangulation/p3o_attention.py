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
from ppomdp.smc.utils import  systematic_resampling

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import TriangulationEnv as env


rng_key = random.PRNGKey(1)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 0.05
tempering = 0.1

learning_rate = 1e-3
batch_size = 256
num_epochs = 250

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


# --- Plot 2: Environment, Trajectories, and Covariance Ellipses ---
from matplotlib import gridspec

plt.style.use('classic')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 10,
    "mathtext.fontset": "cm",
    "svg.fonttype": "none"
})

G_BLUE = '#1A73E8'

cm = 1 / 2.54
fig = plt.figure(figsize=(15 * cm, 6 * cm))  # total: 8 + 4 = 12 cm width
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.4)
ax_env = fig.add_subplot(gs[0])
ax_eig = fig.add_subplot(gs[1])

ts = jnp.arange(env.num_time_steps + 1)

# Plot mean state and belief trajectories
from ppomdp.smc.utils import weighted_mean, weighted_covar
belief_mean_per_traj = weighted_mean(beliefs.particles, beliefs.weights)
belief_covar_per_traj = weighted_covar(beliefs.particles, beliefs.weights)

belief_mean = jnp.mean(belief_mean_per_traj, axis=1)
belief_covar = jnp.mean(belief_covar_per_traj, axis=1)

state_mean = jnp.mean(states, axis=1)
state_center = states - state_mean[:, None, :]
state_covar = jnp.einsum('tnk,tnh->tkh', state_center, state_center) / state_center.shape[1]

x0, y0 = belief_mean[0, 0], belief_mean[0, 2]
x1, y1 = belief_mean[-1, 0], belief_mean[-1, 2]

ax_env.plot(belief_mean[:, 0], belief_mean[:, 2], color=G_BLUE, lw=1.5, zorder=3)
ax_env.scatter(belief_mean[ts[1:-1], 0], belief_mean[ts[1:-1], 2], c=G_BLUE, s=10, zorder=4)
ax_env.scatter(belief_mean[0, 0],  belief_mean[0, 2],  c='g', edgecolors='black', s=10, zorder=4)
ax_env.scatter(belief_mean[-1, 0], belief_mean[-1, 2], c='r', edgecolors='black', s=10, zorder=4)
ax_env.plot([x0, x1], [y0, y1], color='red', linestyle='--', linewidth=1.0, zorder=2)

# ax_env.set_xlim(-225, 25)
# ax_env.set_ylim(-200, 150)
ax_env.set_xlabel(r"$x$")
ax_env.set_ylabel(r"$y$")
ax_env.tick_params(direction='out')
ax_env.tick_params(
    top=False, right=False,      # hide top and right ticks
    bottom=True, left=True       # show bottom and left ticks
)
# ax_env.set_xticks([-200, -150, -100, -50, 0])     # set specific x-axis tick positions
# ax_env.set_yticks([-150, -100, -50, 0, 50, 100])    # set specific y-axis tick positions
ax_env.grid(True)

# --- New subplot: Largest eigenvalue over time ---
eig_max = []
for t in ts:
    cov_2d = belief_covar[t][[0, 2], :][:, [0, 2]]  # project to (0,2) dims
    vals = jnp.linalg.eigvalsh(cov_2d)
    eig_max.append(float(jnp.max(vals)))  # use float() to detach from JAX if needed

ax_eig.plot(ts, eig_max, color='black', lw=1.5)
ax_eig.set_xlabel("Time Step")
ax_eig.set_ylabel("Eigenvalue")
ax_eig.tick_params(direction='out', top=False, right=False)
ax_eig.set_xticks([0, 10, 20, 30])
ax_eig.set_yticks([20, 40, 60, 80])
ax_eig.grid(True)

plt.show()
# plt.savefig("triangulation_trajectory.pdf", bbox_inches='tight')
