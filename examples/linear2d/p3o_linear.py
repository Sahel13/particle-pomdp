import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
jax.config.update("jax_enable_x64", True)

import optax
import time
import matplotlib.pyplot as plt
import pandas as pd

from jax import random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState

from ppomdp.smc import smc, backward_tracing
from ppomdp.policy.arch import LinearGaussDecoder
from ppomdp.policy.linear import create_linear_policy, train_linear_policy
from ppomdp.utils import batch_data, prepare_trajectories, policy_evaluation
from ppomdp.smc.utils import multinomial_resampling, systematic_resampling

from experiments.common import get_pomdp, plot_trajectory
from examples.linear2d.lqg_benchmark import run_lqg_benchmark


print("SMC Linear Policy Training - Linear 2D System")
print("=" * 55)
print("Training particle filter policy on same system as LQG benchmark")
print()

# Get the Linear2D POMDP environment
env = get_pomdp("linear-2d")

print("Environment Configuration:")
print(f"State dimension: {env.state_dim}")
print(f"Action dimension: {env.action_dim}")
print(f"Observation dimension: {env.obs_dim}")
print(f"Time horizon: {env.num_time_steps}")
print()

# Training hyperparameters
rng_key = random.PRNGKey(42)

num_history_particles = 256
num_belief_particles = 256

slew_rate_penalty = 0.001
tempering = 1.0

learning_rate = 1e-3
batch_size = 64
num_epochs = 100

print("Training Configuration:")
print(f"History particles: {num_history_particles}")
print(f"Belief particles: {num_belief_particles}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Epochs: {num_epochs}")
print()

# Create policy components
decoder = LinearGaussDecoder(
    output_dim=env.action_dim,
    init_log_std=constant(jnp.log(1.0)),
)

policy = create_linear_policy(decoder=decoder)

# Initialize policy parameters
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
training_returns = []

print("Starting Training...")
print("-" * 55)

# Training loop
for epoch in range(1, num_epochs + 1):
    start_time = time.time()

    # Evaluate current policy
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
    training_returns.append(avg_return)

    # Run SMC to generate training data
    key, sub_key = random.split(key)
    history_states, belief_states, belief_infos, log_marginal = smc(
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

    # Backward tracing to get training trajectories
    key, sub_key = random.split(key)
    traced_history, traced_belief, _ = backward_tracing(
        sub_key, history_states, belief_states, belief_infos
    )

    # Prepare training data
    key, sub_key = random.split(key)
    actions, particles, weights = prepare_trajectories(
        sub_key, traced_history.actions, traced_belief
    )

    # Train policy on batches
    data_size, _ = actions.shape
    key, sub_key = random.split(key)
    batch_indices = batch_data(sub_key, data_size, batch_size)

    total_loss = 0.0
    for batch_idx in batch_indices:
        action_batch = jax.tree.map(lambda x: x[batch_idx, ...], actions)
        particles_batch = jax.tree.map(lambda x: x[batch_idx, ...], particles)
        weights_batch = jax.tree.map(lambda x: x[batch_idx, ...], weights)

        learner, batch_loss = train_linear_policy(
            policy=policy,
            learner=learner,
            actions=action_batch,
            particles=particles_batch,
            weights=weights_batch,
        )
        total_loss += batch_loss

    # Compute metrics
    entropy = policy.entropy(learner.params)
    end_time = time.time()
    epoch_time = end_time - start_time

    print(
        f"Epoch {epoch:3d}: "
        f"Steps {num_steps:6d} | "
        f"Log marginal {log_marginal:7.3f} | "
        f"Return {avg_return:7.3f} | "
        f"Entropy {entropy:6.3f} | "
        f"Time {epoch_time:5.2f}s"
    )

print("\nTraining Complete!")
print("-" * 55)

# Final evaluation
print("\nFinal Policy Evaluation...")
key, sub_key = random.split(key)
final_rewards, final_states, final_actions, _ = policy_evaluation(
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

final_return = jnp.mean(jnp.sum(final_rewards, axis=0))
return_std = jnp.std(jnp.sum(final_rewards, axis=0))

print(f"Final SMC Policy Performance:")
print(f"  Mean Return: {final_return:.3f}")
print(f"  Std Return:  {return_std:.3f}")
print()

# Run LQG benchmark for comparison
print("Running LQG Benchmark for Comparison...")
lqg_results = run_lqg_benchmark(num_runs=1024, T=env.num_time_steps, seed=42)

print(f"\nPerformance Comparison:")
print(f"  LQG (Optimal):     {-lqg_results['mean_cost']:.3f} ± {lqg_results['sem_cost']:.3f}")
print(f"  SMC Policy:        {final_return:.3f} ± {return_std/jnp.sqrt(1000):.3f}")

optimality_gap = (-lqg_results['mean_cost']) - final_return
print(f"  Optimality Gap:    {optimality_gap:.3f}")
print(f"  Performance Ratio: {final_return/(-lqg_results['mean_cost'])*100:.1f}%")

# Save comparison data to separate CSVs
print("\nSaving benchmark comparison data to separate CSV files...")
lqg_optimal_reward = -lqg_results['mean_cost']

# Create SMC policy DataFrame
smc_data = []
for epoch, smc_reward in enumerate(training_returns, 1):
    smc_data.append({
        'Iteration': epoch,
        'Reward': float(smc_reward)
    })

# Create LQG baseline DataFrame  
lqg_data = []
for epoch in range(1, num_epochs + 1):
    lqg_data.append({
        'Iteration': epoch,
        'Reward': float(lqg_optimal_reward)
    })

# Save to separate CSV files
smc_df = pd.DataFrame(smc_data)
lqg_df = pd.DataFrame(lqg_data)

smc_filename = 'smc_policy_training.csv'
lqg_filename = 'lqg_optimal_baseline.csv'

smc_df.to_csv(smc_filename, index=False)
lqg_df.to_csv(lqg_filename, index=False)

print(f"SMC policy data saved to: {smc_filename} ({len(smc_df)} rows)")
print(f"LQG baseline data saved to: {lqg_filename} ({len(lqg_df)} rows)")

# Plot training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training curve
ax1.plot(range(1, num_epochs + 1), training_returns, 'b-', linewidth=2, label='SMC Policy')
ax1.axhline(y=-lqg_results['mean_cost'], color='r', linestyle='--', linewidth=2, label='LQG Optimal')
ax1.fill_between(range(1, num_epochs + 1),
                 [-lqg_results['mean_cost'] - 1.96*lqg_results['sem_cost']] * num_epochs,
                 [-lqg_results['mean_cost'] + 1.96*lqg_results['sem_cost']] * num_epochs,
                 color='red', alpha=0.2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Average Return')
ax1.set_title('Training Progress')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Sample trajectory
sample_traj_idx = 0
ax2.plot(final_states[:, sample_traj_idx, 0], final_states[:, sample_traj_idx, 1],
         'b-', linewidth=2, label='SMC Policy')
ax2.plot(final_states[0, sample_traj_idx, 0], final_states[0, sample_traj_idx, 1],
         'go', markersize=8, label='Start')
ax2.plot(0, 0, 'ro', markersize=8, label='Target')
ax2.set_xlabel('Position')
ax2.set_ylabel('Velocity')
ax2.set_title('Sample Trajectory (Phase Space)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()

# Plot detailed trajectory
print("\nPlotting detailed trajectory...")
plot_trajectory("linear-2d", final_states[:, 0, :], final_actions[:, 0, :])
