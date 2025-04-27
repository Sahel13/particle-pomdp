import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
import matplotlib.patches as patches
from copy import deepcopy

from ppomdp.envs.pomdps import LightDark2DEnv as env
from ppomdp.envs.pomdps.lightdark2d import stddev_obs


rng_key = random.PRNGKey(1337)

num_history_particles = 256
num_belief_particles = 128
num_target_samples = 512

slew_rate_penalty = 0.01
tempering = 0.5

learning_rate = 3e-4
batch_size = 128
num_epochs = 150

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
    avg_reward, *_ = policy_evaluation(
        rng_key=sub_key,
        env_obj=env,
        policy=policy,
        params=learner.params,
        num_samples=1024,
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
        f"Reward: {avg_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )


eval_params = deepcopy(learner.params)
eval_params["decoder"]["log_std"] = -20.0 * jnp.ones((env.action_dim,))

key, sub_key = random.split(key)
history_states, belief_states, belief_infos, _ = \
    smc(
        rng_key=sub_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        init_prior=env.prior_dist,
        policy_prior=policy,
        policy_prior_params=eval_params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=0.0,
        tempering=0.0,
        history_resample_fn=systematic_resampling,
        belief_resample_fn=multinomial_resampling,
    )

observations = history_states.particles.observations
actions = history_states.particles.actions
state_means = belief_infos.mean
state_covars = belief_infos.covar

fig, axs = plt.subplots(4, 1, figsize=(8, 8))
for n in range(num_history_particles):
    axs[0].plot(state_means[:, n, 0])
    axs[0].set_ylabel('State-1')

    axs[1].plot(state_means[:, n, 1])
    axs[1].set_ylabel('State-2')

    axs[2].plot(actions[:, n, 0])
    axs[2].set_ylabel('Act-1')

    axs[3].plot(actions[:, n, 1])
    axs[3].set_ylabel('Act-2')

    # axs[4].plot(state_covars[:, n, 0, 0])
    # axs[4].set_ylabel('Var-1')
    #
    # axs[5].plot(state_covars[:, n, 1, 1])
    # axs[5].set_xlabel('Time Step')
    # axs[5].set_ylabel('Var-2')

plt.tight_layout()
plt.show()

# plot environment
xgrid = jnp.linspace(-1.0, 6.0, 100)
ygrid = jnp.linspace(-2.0, 2.5, 100)
X, Y = jnp.meshgrid(xgrid, ygrid)

Z = jnp.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xy = jnp.array([X[i, j], Y[i, j], 0.0, 0.0])
        Z = Z.at[i, j].set(
            jnp.linalg.norm(stddev_obs(xy))
        )

plt.imshow(-Z, extent=(-1.0, 6.0, -2.0, 2.5), origin='lower', cmap='gray')
plt.title('Light-Dark Environment')
plt.xlabel('X')
plt.ylabel('Y')

# plot means and covars
plt.plot(
    jnp.mean(state_means, axis=1)[:, 0],
    jnp.mean(state_means, axis=1)[:, 1], 'r-'
)

for t in range(0, env.num_time_steps + 1, 2):
    mean = jnp.mean(state_means, axis=1)[t, :2]
    covar = jnp.mean(state_covars, axis=1)[t, :2, :2]

    eigvals, eigvecs = jnp.linalg.eigh(covar)
    angle = jnp.degrees(jnp.arctan2(eigvecs[0, 1], eigvecs[0, 0]))
    width, height = jnp.sqrt(eigvals)
    ell = patches.Ellipse(
        mean, width, height, angle=angle, edgecolor='b', facecolor='none'
    )
    plt.gca().add_patch(ell)

plt.show()
