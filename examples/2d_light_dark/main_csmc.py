import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import jax
from jax import random
import jax.numpy as jnp

from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling
from ppomdp.csmc import csmc
from ppomdp.core import Reference
from ppomdp.bijector import Tanh
from ppomdp.policy import GRU, get_recurrent_policy, train_step
from ppomdp.utils import batch_data

from distrax import Chain, ScalarAffine
from flax.linen.initializers import constant
from flax.training.train_state import TrainState

from copy import deepcopy
import optax

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from environment import prior_dist, trans_model, obs_model, reward_fn
from environment import action_dim, obs_dim, num_time_steps, stddev_obs

jax.config.update("jax_enable_x64", True)

init_log_std = jnp.log(jnp.array([1.0, 1.0]))

network = GRU(
    dim=action_dim,
    feature_fn=lambda x: x,
    encoder_size=[256, 256],
    recurr_size=[64, 64],
    output_size=[256, 256],
    init_log_std=constant(init_log_std),
)

shift = jnp.zeros((action_dim,))
scale = jnp.array([100.0, 100.0])
bijector = Chain([ScalarAffine(shift, scale), Tanh()])
policy = get_recurrent_policy(network, bijector)

rng_key = random.PRNGKey(1234)

num_outer_particles = 512
num_inner_particles = 256
tempering = 0.01
slew_rate_penalty = 1e-3

learning_rate = 3e-4
batch_size = 64
num_epochs = 1000

# Initialize training state
key, obs_key, param_key = random.split(rng_key, 3)
init_carry = policy.reset(num_outer_particles)
init_obs = random.normal(obs_key, (num_outer_particles, obs_dim))
init_params = network.init(param_key, init_carry, init_obs)["params"]
train_state = TrainState.create(
    apply_fn=network.apply,
    params=init_params,
    tx=optax.adam(learning_rate)
)

jitted_smc = jax.jit(smc, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9))
jitted_csmc = jax.jit(csmc, static_argnums=(1, 2, 3, 4, 5, 6, 7, 9))
jitted_backward_tracing = jax.jit(backward_tracing, static_argnums=(5,))
jitted_mcmc_backward_sampling = \
    jax.jit(mcmc_backward_sampling, static_argnums=(1, 4, 5, 6, 8, 9, 10))

# run init nested smc
key, sub_key = random.split(key)
outer_states, inner_states, inner_info, log_marginal = \
    jitted_smc(
        sub_key,
        num_time_steps,
        num_outer_particles,
        num_inner_particles,
        prior_dist,
        trans_model,
        obs_model,
        policy,
        train_state.params,
        reward_fn,
        tempering,
        slew_rate_penalty
    )

# # trace ancestors of outer states
# key, sub_key = random.split(key)
# traced_outer, traced_inner, _ = \
#     jitted_backward_tracing(sub_key, outer_states, inner_states, inner_info)

# backward sample outer states
key, sub_key = random.split(key)
traced_outer, traced_inner = jitted_mcmc_backward_sampling(
    sub_key, num_outer_particles, outer_states, inner_states, trans_model, obs_model,
    policy, train_state.params, reward_fn, tempering, slew_rate_penalty
)

# sample a new reference
key, sub_key = random.split(key)
idx = jax.random.choice(sub_key, jnp.arange(num_outer_particles))
outer_reference = jax.tree.map(lambda x: x[:, idx], traced_outer.particles)
inner_reference = jax.tree.map(lambda x: x[:, idx], traced_inner)
reference = Reference(
    outer_particles=outer_reference,
    inner_state=inner_reference
)

for i in range(1, num_epochs + 1):
    start_time = time.time()

    # run nested conditional smc
    key, sub_key = random.split(key)
    outer_states, inner_states, inner_info, log_marginal = \
        jitted_csmc(
            sub_key,
            num_time_steps,
            num_outer_particles,
            num_inner_particles,
            prior_dist,
            trans_model,
            obs_model,
            policy,
            train_state.params,
            reward_fn,
            tempering,
            slew_rate_penalty,
            reference
        )

    # # trace ancestors of outer states
    # key, sub_key = random.split(key)
    # traced_outer, traced_inner, _ = \
    #     jitted_backward_tracing(sub_key, outer_states, inner_states, inner_info)

    # backward sample outer states
    key, sub_key = random.split(key)
    traced_outer, traced_inner = jitted_mcmc_backward_sampling(
        sub_key, num_outer_particles, outer_states, inner_states, trans_model, obs_model,
        policy, train_state.params, reward_fn, tempering, slew_rate_penalty
    )

    # sample a new reference
    key, sub_key = random.split(key)
    idx = jax.random.choice(sub_key, jnp.arange(num_outer_particles))
    outer_reference = jax.tree.map(lambda x: x[:, idx], traced_outer.particles)
    inner_reference = jax.tree.map(lambda x: x[:, idx], traced_inner)
    reference = Reference(
        outer_particles=outer_reference,
        inner_state=inner_reference
    )

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_outer_particles, batch_size)
    for batch_idx in batch_indices:
        outer_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_outer.particles)
        train_state, batch_loss = train_step(policy, train_state, outer_batch)
        loss += batch_loss

    entropy = policy.entropy(train_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Log marginal: {log_marginal:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

eval_state = deepcopy(train_state)
eval_state.params["log_std"] = -20.0 * jnp.ones((action_dim,))

key, sub_key = random.split(key)
outer_states, inner_states, inner_infos, _ = \
    jitted_smc(
        sub_key,
        num_time_steps,
        num_outer_particles,
        num_inner_particles,
        prior_dist,
        trans_model,
        obs_model,
        policy,
        eval_state.params,
        reward_fn,
        tempering=0.0,
        slew_rate_penalty=0.0,
    )

# trace ancestors of outer states
key, sub_key = random.split(key)
outer_states, inner_states, inner_infos = \
    jitted_backward_tracing(sub_key, outer_states, inner_states, inner_infos)

observations = outer_states.particles[0]
actions = outer_states.particles[1]
state_means = inner_infos.mean
state_covars = inner_infos.covar
rwrds = outer_states.rewards

fig, axs = plt.subplots(4, 1, figsize=(8, 8))
for n in range(num_outer_particles):
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

for t in range(0, num_time_steps + 1, 2):
    mean = jnp.mean(state_means, axis=1)[t, :2]
    covar = jnp.mean(state_covars, axis=1)[t, :2, :2]

    eigvals, eigvecs = jnp.linalg.eigh(covar)
    angle = jnp.degrees(jnp.arctan2(eigvecs[0, 1], eigvecs[0, 0]))
    width, height = jnp.sqrt(eigvals)
    ell = patches.Ellipse(
        mean, width, height,
        angle=angle, edgecolor='b', facecolor='none'
    )
    plt.gca().add_patch(ell)


# plot realization
states = []
actions = []
observations = []

key, state_key, obs_key = random.split(key, 3)

state = jnp.array([2., 2., 0., 0.])
obs = obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, obs, carry, eval_state.params)
    state = trans_model.sample(state_key, state, action[0])
    obs = obs_model.sample(obs_key, state)

    states.append(state)
    actions.append(action[0])
    observations.append(obs)

states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))
observations = jnp.squeeze(jnp.array(observations))

plt.plot(states[:, 0], states[:, 1], 'g-')
plt.show()
