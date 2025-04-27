import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import jax
jax.config.update("jax_enable_x64", True)

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block

from ppomdp.core import Reference
from ppomdp.smc._smc import smc, backward_tracing
from ppomdp.smc._csmc import csmc

from ppomdp.bijector import Tanh
from ppomdp.utils import batch_data, policy_evaluation
from ppomdp.policy.arch import GRUEncoder, MLPDecoder
from ppomdp.policy.gauss import (
    RecurrentNeuralGaussPolicy,
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_pathwise
)

import time
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ppomdp.envs.pomdps import LightDark2DEnv as env
from ppomdp.envs.pomdps.lightdark2d import stddev_obs


rng_key = random.PRNGKey(1337)

num_history_particles = 128
num_belief_particles = 32

slew_rate_penalty = 5e-4
tempering = 0.5
num_moves = 1

learning_rate = 3e-4
batch_size = 256
num_epochs = 100

encoder = GRUEncoder(
    feature_fn=lambda x: x,
    encoder_size=(256, 256),
    recurr_size=(128, 128),
)
decoder = MLPDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
)
network = RecurrentNeuralGaussPolicy(
    encoder=encoder,
    decoder=decoder,
    init_log_std=constant(jnp.log(2.0)),
)
bijector = Block(Tanh(), ndims=1)

key, sub_key = random.split(rng_key, 2)
policy = create_recurrent_neural_gauss_policy(network, bijector)
train_state = policy.init(
    rng_key=sub_key,
    input_dim=env.obs_dim,
    output_dim=env.action_dim,
    batch_dim=num_history_particles,
    learning_rate=learning_rate
)

# run init nested smc
key, sub_key = random.split(key)
history_states, belief_states, belief_infos, _ = \
    smc(sub_key, env.num_time_steps, num_history_particles, num_belief_particles, env.prior_dist, policy,
        train_state.params, env.trans_model, env.obs_model, env.reward_fn, tempering, slew_rate_penalty)

# trace ancestors of history states
key, sub_key = random.split(key)
traced_history, traced_belief, _ = (
    backward_tracing(sub_key, history_states, belief_states, belief_infos))

# # backward sample history states
# key, sub_key = random.split(key)
# traced_history, traced_belief = mcmc_backward_sampling(
#     sub_key,
#     num_history_particles,
#     history_states,
#     belief_states,
#     env.trans_model,
#     policy,
#     train_state.params,
#     env.reward_fn,
#     tempering,
#     slew_rate_penalty
# )

# sample a new reference
key, sub_key = random.split(key)
idx = jax.random.choice(sub_key, jnp.arange(num_history_particles))
reference = Reference(
    history_particles=jax.tree.map(lambda x: x[:, idx], traced_history),
    belief_state=jax.tree.map(lambda x: x[:, idx], traced_belief)
)

for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current (deterministic) policy
    key, sub_key = random.split(key)
    expected_reward, *_ = \
        policy_evaluation(sub_key, env, policy, train_state.params)

    for _ in range(num_moves):
        # run nested conditional smc
        key, sub_key = random.split(key)
        history_states, belief_states, belief_infos, log_marginal = \
            csmc(
                sub_key,
                env.num_time_steps,
                num_history_particles,
                num_belief_particles,
                env.prior_dist,
                env.trans_model,
                env.obs_model,
                policy,
                train_state.params,
                env.reward_fn,
                tempering,
                slew_rate_penalty,
                reference
            )

        # trace ancestors of history states
        key, sub_key = random.split(key)
        traced_history, traced_belief, _ = \
            backward_tracing(sub_key, history_states, belief_states, belief_infos)

        # # backward sample history states
        # key, sub_key = random.split(key)
        # traced_history, traced_belief = mcmc_backward_sampling(
        #     sub_key,
        #     num_history_particles,
        #     history_states,
        #     belief_states,
        #     env.trans_model,
        #     policy,
        #     train_state.params,
        #     env.reward_fn,
        #     tempering,
        #     slew_rate_penalty
        # )

        # sample a new reference
        key, sub_key = random.split(key)
        idx = jax.random.choice(sub_key, jnp.arange(num_history_particles))
        reference = Reference(
            history_particles=jax.tree.map(lambda x: x[:, idx], traced_history),
            belief_state=jax.tree.map(lambda x: x[:, idx], traced_belief)
        )

    # update policy parameters
    loss = 0.0
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, num_history_particles, batch_size)
    for batch_idx in batch_indices:
        history_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history)
        train_state, batch_loss = \
            train_recurrent_neural_gauss_policy_pathwise(policy, train_state, history_batch)
        loss += batch_loss

    entropy = policy.entropy(train_state.params)
    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Reward: {expected_reward:.3f}, "
        f"Entropy: {entropy:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

eval_state = deepcopy(train_state)
eval_state.params["log_std"] = -20.0 * jnp.ones((env.action_dim,))

key, sub_key = random.split(key)
history_states, belief_states, belief_infos, _ = \
    smc(sub_key, env.num_time_steps, num_history_particles, num_belief_particles, env.prior_dist, policy,
        eval_state.params, env.trans_model, env.obs_model, env.reward_fn, slew_rate_penalty=0.0, tempering=0.0)

observations = history_states.particles.observations
actions = history_states.particles.actions
state_means = belief_infos.mean
state_covars = belief_infos.covar
rwrds = history_states.rewards

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


# plot realization
states = []
actions = []
observations = []

key, state_key, obs_key = random.split(key, 3)

state = jnp.array([2., 2., 0., 0.])
obs = env.obs_model.sample(obs_key, state)
carry = policy.reset(1)

states.append(state)
observations.append(obs)

for _ in range(env.num_time_steps):
    key, state_key, obs_key, action_key = random.split(key, 4)

    carry, action = policy.sample(action_key, carry, obs, eval_state.params)
    state = env.trans_model.sample(state_key, state, action[0])
    obs = env.obs_model.sample(obs_key, state)

    states.append(state)
    actions.append(action[0])
    observations.append(obs)

states = jnp.squeeze(jnp.array(states))
actions = jnp.squeeze(jnp.array(actions))
observations = jnp.squeeze(jnp.array(observations))

plt.plot(states[:, 0], states[:, 1], 'g-')
plt.show()
