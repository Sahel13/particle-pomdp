import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
from ppomdp.utils import batch_data, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import LightDark2DEnv as env


rng_key = random.PRNGKey(1337)

num_history_particles = 256
num_belief_particles = 32
num_target_samples = 512

slew_rate_penalty = 0.0
tempering = 0.2

learning_rate = 3e-4
batch_size = 64
num_epochs = 250

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


# --- Plot: Environment, Trajectories, and Covariance Ellipses ---
from ppomdp.envs.pomdps.lightdark2d import stddev_obs
from matplotlib.patches import Ellipse

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

# Helper function to plot covariance ellipse
def plot_covar_ellipse(ax, mean, covar, color):
    """Calculates and plots a covariance ellipse for 2D data."""
    eigvals, eigvecs = jnp.linalg.eigh(covar)
    angle = jnp.degrees(jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = jnp.sqrt(jnp.maximum(eigvals, 1e-9)) # Use std dev for ellipse size
    ell = Ellipse(mean, width, height, angle=angle, edgecolor=color, facecolor='none', linewidth=1.)
    ax.add_patch(ell)

# Plot environment background (observation noise level)
xgrid = jnp.linspace(-1.0, 6.0, 100)
ygrid = jnp.linspace(-1.0, 2.5, 100)
X, Y = jnp.meshgrid(xgrid, ygrid)

light_level = jnp.zeros_like(X)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        _state = jnp.array([X[r, c], Y[r, c]] + [0.0] * (env.state_dim - 2))
        _light_level = jnp.linalg.norm(stddev_obs(_state[:env.state_dim]))
        light_level = light_level.at[r, c].set(_light_level)


cm = 1 / 2.54
fig_env, ax_env = plt.subplots(figsize=(8 * cm, 6 * cm))  # 8cm x 6cm

ts = [0, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

im = ax_env.imshow(
    -light_level,
    extent=(xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()),
    origin='lower', cmap='gray', aspect='auto', vmax=1.
)

# Plot mean state and belief trajectories
from ppomdp.smc.utils import weighted_mean, weighted_covar
belief_mean_per_traj = weighted_mean(beliefs.particles, beliefs.weights)
belief_covar_per_traj = weighted_covar(beliefs.particles, beliefs.weights)

belief_mean = jnp.mean(belief_mean_per_traj, axis=1)
belief_covar = jnp.mean(belief_covar_per_traj, axis=1)

state_mean = jnp.mean(states, axis=1)
state_center = states - state_mean[:, None, :]
state_covar = jnp.einsum('tnk,tnh->tkh', state_center, state_center) / state_center.shape[1]

x0, y0 = belief_mean[0, 0], belief_mean[0, 1]
x1, y1 = belief_mean[-1, 0], belief_mean[-1, 1]

ax_env.plot(belief_mean[:, 0], belief_mean[:, 1], color=G_BLUE, lw=1.5, zorder=3)
ax_env.scatter(belief_mean[ts[1:-1], 0], belief_mean[ts[1:-1], 1], c=G_BLUE, s=10, zorder=4)
ax_env.scatter(belief_mean[0, 0],  belief_mean[0, 1],  c='g', edgecolors='white', s=10, zorder=4)
ax_env.scatter(belief_mean[-1, 0], belief_mean[-1, 1], c='r', edgecolors='white', s=10, zorder=4)
ax_env.plot([x0, x1], [y0, y1], color='red', linestyle='--', linewidth=1.0, zorder=2)

# Plot covariance ellipses at intervals
for t in ts:
    plot_covar_ellipse(ax_env, belief_mean[t, :2], belief_covar[t, :2, :2], 'white')

ax_env.set_xlabel(r"$x$")
ax_env.set_ylabel(r"$y$")
ax_env.tick_params(direction='out')
ax_env.tick_params(
    top=False, right=False,      # hide top and right ticks
    bottom=True, left=True       # show bottom and left ticks
)
ax_env.set_xticks([0, 1, 2, 3, 4 ,5])     # set specific x-axis tick positions
ax_env.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])    # set specific y-axis tick positions
plt.show()
# plt.savefig("light_dark_trajectory.pdf", bbox_inches='tight')
