import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import jax
# jax.config.update("jax_enable_x64", True)

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from distrax import Block, MultivariateNormalDiag

from ppomdp.core import Reference
from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling
from ppomdp.csmc import csmc

from ppomdp.bijector import Tanh
from ppomdp.utils import batch_data
from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.gauss import (
    RecurrentNeuralGauss,
    create_recurrent_gauss_policy,
    train_recurrent_gauss_policy
)

import time
from copy import deepcopy
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import LightDark1DEnv as env


rng_key = random.PRNGKey(123)

num_history_particles = 256
num_belief_particles = 256

slew_rate_penalty = 0.001
tempering = 5.
num_moves = 5

learning_rate = 3e-4
batch_size = 256
num_epochs = 500

encoder = GRUEncoder(
    feature_fn=lambda x: x,
    encoder_size=(256, 256),
    recurr_size=(64, 64),
)

decoder = MLPDecoder(
    decoder_size=(256, 256),
    output_dim=env.action_dim,
)

network = RecurrentNeuralGauss(
    encoder=encoder,
    decoder=decoder,
    init_log_std=constant(jnp.log(1.0)),
)

bijector = Block(Tanh(), ndims=1)

key, sub_key = random.split(rng_key, 2)
policy = create_recurrent_gauss_policy(network, bijector)
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
    smc(
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

for i in range(1, num_epochs + 1):
    start_time = time.time()

    # evaluate current policy
    key, sub_key = random.split(key)
    history_states, _, _, _ = \
        smc(
            sub_key,
            env.num_time_steps,
            int(4 * num_history_particles),
            int(4 * num_belief_particles),
            env.prior_dist,
            env.trans_model,
            env.obs_model,
            policy,
            train_state.params,
            env.reward_fn,
            tempering=0.0,
            slew_rate_penalty=0.0,
        )
    expected_reward = jnp.mean(jnp.sum(history_states.rewards, axis=0))

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
        train_state, batch_loss = train_recurrent_gauss_policy(policy, train_state, history_batch)
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
    smc(
        sub_key,
        env.num_time_steps,
        num_history_particles,
        num_belief_particles,
        env.prior_dist,
        env.trans_model,
        env.obs_model,
        policy,
        eval_state.params,
        env.reward_fn,
        tempering=0.0,
        slew_rate_penalty=0.0,
    )

observations = history_states.particles.observations
actions = history_states.particles.actions
state_means = belief_infos.mean
state_covars = belief_infos.covar

fig, axs = plt.subplots(4, 1, figsize=(10, 8))
for n in range(num_history_particles):
    axs[0].plot(state_means[:, n, :])
    axs[0].set_title('Mean over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('State')

    axs[1].plot(state_covars[:, n, 0, 0])
    axs[1].set_title('Variance over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Variance')

    axs[2].plot(actions[:, n, :])
    axs[2].set_title('Action over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Action')

    axs[3].plot(observations[:, n, :])
    axs[3].set_title('Observation over Time')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Observation')

plt.tight_layout()
plt.show()


states = []
actions = []
observations = []

key, state_key, obs_key = random.split(key, 3)
init_dist = MultivariateNormalDiag(
    loc=1.0 * jnp.ones((env.state_dim,)),
    scale_diag=0.25 * jnp.ones((env.state_dim,))
)

state = init_dist.sample(seed=state_key)
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

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(states)
axs[0].set_ylabel("States")

axs[1].plot(actions)
axs[1].set_ylabel("Action")

plt.tight_layout()
plt.show()
