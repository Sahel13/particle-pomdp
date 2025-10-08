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
    train_recurrent_neural_gauss_policy_pathwise,
)
from ppomdp.utils import batch_data, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import TriangulationEnv as env


rng_key = random.PRNGKey(0)

num_history_particles = 128
num_belief_particles = 32
num_target_samples = 512

slew_rate_penalty = 0.05
tempering = 0.1

learning_rate = 1e-3
batch_size = 128
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
        stochastic=True
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