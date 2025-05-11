import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
# jax.config.update("jax_enable_x64", True)

import optax

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.smc import smc, backward_tracing, mcmc_backward_sampling

from ppomdp.bijector import Tanh
from ppomdp.policy.arch import AttentionEncoder, NeuralGaussDecoder
from ppomdp.policy.attention import (
    create_attention_policy,
    train_attention_policy
)
from ppomdp.utils import batch_data, custom_split, policy_evaluation, policy_evaluation_with_beliefs
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

import time
import matplotlib.pyplot as plt

from ppomdp.envs.pomdps import TargetEnv as env


rng_key = random.PRNGKey(1337)

num_history_particles = 128
num_belief_particles = 32
num_target_samples = 256

slew_rate_penalty = 0.05
tempering = 0.1

learning_rate = 1e-3
batch_size = 256
num_epochs = 500

bijector = Block(Tanh(), ndims=1)
encoder = AttentionEncoder(
    hidden_dim=256,
    output_dim=256,
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
    rewards, *_ = policy_evaluation_with_beliefs(
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

    # trace ancestors of history states
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = \
        backward_tracing(sub_key, history_states, belief_states, belief_infos)

    # # backward sample history states
    # key, sub_key = random.split(key)
    # traced_history, traced_belief = mcmc_backward_sampling(
    #     rng_key=sub_key,
    #     num_samples=num_target_samples,
    #     policy_prior=policy,
    #     policy_prior_params=learner.params,
    #     trans_model=env.trans_model,
    #     reward_fn=env.reward_fn,
    #     slew_rate_penalty=slew_rate_penalty,
    #     tempering=tempering,
    #     history_states=history_states,
    #     belief_states=belief_states,
    # )
    #
    # # update policy parameters
    # loss = 0.0
    # key, sub_key = random.split(key)
    # batch_indices = batch_data(sub_key, num_target_samples, batch_size)
    # for batch_idx in batch_indices:
    #     action_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.actions)
    #     observation_batch = jax.tree.map(lambda x: x[:, batch_idx], traced_history.observations)
    #
    #     learner, batch_loss = train_recurrent_neural_gauss_policy_pathwise(
    #         policy=policy,
    #         learner=learner,
    #         actions=action_batch,
    #         observations=observation_batch,
    #     )
    #     loss += batch_loss

    loss = 0.0
    key, sub_key = random.split(key)

    num_time_steps_p1, _, _ = traced_history.actions.shape
    data_size = (num_time_steps_p1 - 1) * num_history_particles


    from ppomdp.smc.utils import resample_belief
    def _resample_belief(key, belief):
        key, sub_keys = custom_split(key, num_history_particles + 1)
        return jax.vmap(resample_belief, in_axes=(0, 0, None))(
            sub_keys, belief, systematic_resampling
        )

    key, sub_keys = custom_split(key, env.num_time_steps + 2)
    traced_belief = jax.vmap(_resample_belief, in_axes=(0, 0))(
        sub_keys, traced_belief
    )

    actions = traced_history.actions[1:].reshape((-1, traced_history.actions.shape[-1]))
    particles = traced_belief.particles[:-1].reshape((-1, num_belief_particles, traced_belief.particles.shape[-1]))
    weights = traced_belief.weights[:-1].reshape((-1, traced_belief.weights.shape[-1]))

    batch_indices = batch_data(sub_key, data_size, batch_size)
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
_, states, actions, beliefs = policy_evaluation_with_beliefs(
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

plt.figure()
plt.title("Simulated trajectory")

# Plot trajectories
plt.plot(states[..., 0], states[..., 2], alpha=0.7)

# Plot start and target points
plt.scatter(-200, 100, color="black", s=100)
plt.scatter(0, 0, color="orange", s=100)

# Plot direct path line
plt.plot([-200, 0], [100, 0], "r--")

# Configure plot
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True)
plt.tight_layout()
plt.show()
