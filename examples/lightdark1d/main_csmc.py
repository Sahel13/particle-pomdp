import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import jax
# jax.config.update("jax_enable_x64", True)

import optax

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.core import Reference
from ppomdp.smc import smc, csmc, backward_tracing, mcmc_backward_sampling
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, NeuralGaussDecoder
from ppomdp.policy.gauss import (
    create_recurrent_neural_gauss_policy,
    train_recurrent_neural_gauss_policy_pathwise,
)
from ppomdp.utils import batch_data, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling

import time
from copy import deepcopy
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import LightDark1DEnv as env


rng_key = random.PRNGKey(123)

num_history_particles = 256
num_belief_particles = 32
num_target_samples = 512

slew_rate_penalty = 0.0015
tempering = 10.
num_moves = 1

learning_rate = 1e-4
batch_size = 64
num_epochs = 100

bijector = Block(Tanh(), ndims=1)
encoder = GRUEncoder(
    feature_fn=lambda x: x,
    dense_sizes=(256, 256),
    recurr_sizes=(64, 64),
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
    apply_fn=None,
    params=params,
    tx=optax.adam(learning_rate)
)

# run init nested smc
key, sub_key = random.split(key)
history_states, belief_states, belief_infos, _ = \
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
        history_resample_fn=multinomial_resampling,
        belief_resample_fn=multinomial_resampling,
    )

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

# sample a new reference
key, sub_key = random.split(key)
idx = jax.random.choice(sub_key, jnp.arange(num_target_samples))
reference = Reference(
    history_particles=jax.tree.map(lambda x: x[:, idx], traced_history),
    belief_state=jax.tree.map(lambda x: x[:, idx], traced_belief)
)

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
        stochastic=True
    )
    avg_return = jnp.mean(jnp.sum(rewards, axis=0))

    for _ in range(num_moves):
        # run nested conditional smc
        key, sub_key = random.split(key)
        history_states, belief_states, belief_infos, log_marginal = \
            csmc(
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
                reference=reference,
                history_resample_fn=multinomial_resampling,
                belief_resample_fn=multinomial_resampling,
            )

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

        # sample a new reference
        key, sub_key = random.split(key)
        idx = jax.random.choice(sub_key, jnp.arange(num_target_samples))
        reference = Reference(
            history_particles=jax.tree.map(lambda x: x[:, idx], traced_history),
            belief_state=jax.tree.map(lambda x: x[:, idx], traced_belief)
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

    end_time = time.time()
    time_diff = end_time - start_time

    print(
        f"Epoch: {i:3d}, "
        f"Reward: {avg_return:.3f}, "
        f"Time per epoch: {time_diff:.3f}s"
    )

evaluator = deepcopy(learner)
evaluator.params["decoder"]["log_std"] = -20.0 * jnp.ones((env.action_dim,))

key, sub_key = random.split(key)
history_states, belief_states, belief_infos, _ = \
    smc(
        rng_key=sub_key,
        num_time_steps=env.num_time_steps,
        num_history_particles=num_history_particles,
        num_belief_particles=num_belief_particles,
        belief_prior=env.belief_prior,
        policy_prior=policy,
        policy_prior_params=evaluator.params,
        trans_model=env.trans_model,
        obs_model=env.obs_model,
        reward_fn=env.reward_fn,
        slew_rate_penalty=0.0,
        tempering=0.0,
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

key, sub_key = random.split(key)
_, states, actions = policy_evaluation(
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

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(states[..., 0])
axs[0].set_ylabel("States")

axs[1].plot(actions[..., 0])
axs[1].set_ylabel("Action")

plt.tight_layout()
plt.show()
