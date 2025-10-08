import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax import random, numpy as jnp
from distrax import MultivariateNormalDiag

from ppomdp.lqg import create_lqg_policy, lqg_policy_evaluation
from ppomdp.envs.pomdps.linear2d import LINEAR_2D_MATRICES, reward_fn


def get_system_from_pomdp_env():
    """Get system matrices from the Linear2D POMDP environment.

    Returns:
        System matrices and parameters for consistent benchmarking
    """
    # Extract matrices from the POMDP environment
    A = LINEAR_2D_MATRICES['A']
    B = LINEAR_2D_MATRICES['B']
    C = LINEAR_2D_MATRICES['C']
    Q_dyn = LINEAR_2D_MATRICES['Q_process']
    R_obs = LINEAR_2D_MATRICES['R_obs']
    Q_cost = LINEAR_2D_MATRICES['Q_cost']
    R_cost = LINEAR_2D_MATRICES['R_cost']
    dt = LINEAR_2D_MATRICES['dt']

    # Extract initial covariance from system matrices
    init_scale_diag = LINEAR_2D_MATRICES['init_scale_diag']
    initial_cov = jnp.diag(init_scale_diag**2)

    return A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_cov


def generate_random_initial_states(num_runs: int, seed: int = 42):
    """Generate random initial conditions using the same distribution as POMDP environment.

    Args:
        num_runs: Number of initial conditions to generate
        seed: Random seed for reproducibility

    Returns:
        Array of initial states (num_runs, 2)
    """
    key = random.key(seed)

    # Create the same initial distribution as the POMDP environment
    init_loc = LINEAR_2D_MATRICES['init_loc']
    init_scale_diag = LINEAR_2D_MATRICES['init_scale_diag']
    init_dist = MultivariateNormalDiag(
        loc=init_loc,  # Use same initial location as POMDP environment
        scale_diag=init_scale_diag
    )

    # Generate initial conditions by sampling from the POMDP's init_dist
    initial_states = []

    for i in range(num_runs):
        key, subkey = random.split(key)
        initial_state = init_dist.sample(seed=subkey)
        initial_states.append(initial_state)

    return jnp.array(initial_states)


def run_lqg_trial(
    initial_state: jnp.ndarray,
    system_matrices: tuple,
    T: int = 25,
    seed: int = 42
) -> dict:
    """Run a single LQG trial with given initial condition.

    Args:
        initial_state: Initial state [position, velocity]
        system_matrices: Tuple of (A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_cov)
        T: Number of time steps
        seed: Random seed for noise

    Returns:
        Dictionary with trial results
    """
    A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_cov = system_matrices

    # Create LQG policy
    lqg_policy = create_lqg_policy(
        A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_state, initial_cov
    )

    # Debug: Print LQR gain on first trial only
    if seed == 42:  # Only print on first trial
        print(f"DEBUG: LQR Gain Matrix K = {lqg_policy.lqr.K}")
        print(f"DEBUG: Initial belief mean = {lqg_policy.belief.mean}")
        print(f"DEBUG: System matrices check:")
        print(f"  A = \n{A}")
        print(f"  Q_cost = \n{Q_cost}")
        print(f"  R_cost = {R_cost.item()}")

    # Simulation
    key = random.key(seed)

    true_states = jnp.zeros((T + 1, 2))
    observations = jnp.zeros((T, 1))
    actions = jnp.zeros((T, 1))
    costs = jnp.zeros(T)

    # Initialize
    true_states = true_states.at[0].set(initial_state)
    lqg_policy.reset()

    # Simulation loop
    for t in range(T):
        # Get noisy observation
        key, subkey = random.split(key)
        obs_noise = random.multivariate_normal(subkey, jnp.zeros(1), R_obs)
        observation = C @ true_states[t] + obs_noise
        observations = observations.at[t].set(observation)

        # Get action from LQG policy
        action = lqg_policy.get_action(observation)
        actions = actions.at[t].set(action)

        # Compute instantaneous cost: 0.5 * (x^T Q x + u^T R u) to match POMDP reward
        state_cost = true_states[t].T @ Q_cost @ true_states[t]
        control_cost = action.T @ R_cost @ action
        total_cost = 0.5 * (state_cost + control_cost.item())  # Match POMDP scaling
        costs = costs.at[t].set(total_cost)

        # Simulate true dynamics with process noise
        key, subkey = random.split(key)
        process_noise = random.multivariate_normal(subkey, jnp.zeros(2), Q_dyn)
        next_state = A @ true_states[t] + B @ action.reshape(-1) + process_noise
        true_states = true_states.at[t + 1].set(next_state)

        # Update LQG belief for next iteration
        lqg_policy.belief = lqg_policy.kf.predict(lqg_policy.belief, action)

    # Compute metrics
    cumulative_cost = jnp.sum(costs)
    final_position_error = abs(true_states[-1, 0])
    final_velocity_error = abs(true_states[-1, 1])

    return {
        'cumulative_cost': cumulative_cost,
        'final_position_error': final_position_error,
        'final_velocity_error': final_velocity_error,
        'final_state': true_states[-1],
        'initial_state': initial_state
    }


def run_lqg_benchmark(num_runs: int = 10, T: int = 25, seed: int = 42):
    """Run LQG benchmark using policy evaluation.

    This benchmark mirrors the exact structure of SMC policy evaluation to ensure
    fair comparison between LQG (optimal) and particle filter methods.

    Args:
        num_runs: Number of trajectory samples (like num_trajectory_samples in SMC)
        T: Number of time steps (like num_time_steps in SMC)
        seed: Random seed for reproducibility

    Returns:
        Benchmark results and statistics
    """
    print("LQG Controller Benchmark")
    print("=" * 45)
    print(f"Linear2D POMDP System | {num_runs} trajectories | T={T} steps")
    print("Mirroring SMC policy_evaluation structure for direct comparison")
    print()

    # Get system from POMDP environment
    system_matrices = get_system_from_pomdp_env()
    A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_cov = system_matrices

    print("System Configuration:")
    print(f"State transition A:\n{A}")
    print(f"Control matrix B:\n{B.flatten()}")
    print(f"Observation matrix C: {C.flatten()}")
    print(f"Cost weights Q_diag: {jnp.diag(Q_cost)}")
    print(f"Control penalty R: {R_cost.item()}")
    print()

    # DEBUG: Check LQR controller
    from ppomdp.lqg import solve_lqr
    lqr_controller = solve_lqr(A, B, Q_cost, R_cost)
    print("DEBUG: LQR Controller:")
    print(f"  LQR gain K: {lqr_controller.K.flatten()}")
    print(f"  Cost-to-go P:\n{lqr_controller.P}")

    # DEBUG: Quick sanity check on manual cost calculation
    test_state = jnp.array([1., 0.])
    test_action = jnp.array([0.])
    manual_cost = 0.5 * (test_state.T @ Q_cost @ test_state + test_action.T @ R_cost @ test_action)
    manual_reward = reward_fn(test_state, test_action, 1)
    print(f"\nDEBUG: Manual cost calculation for state [1,0], action [0]:")
    print(f"  Manual cost: {manual_cost}")
    print(f"  Reward function: {manual_reward}")
    print(f"  Should be: manual_cost = 0.5, reward = -0.5")
    print()

    # Run LQG policy evaluation
    key = random.key(seed)
    rewards, states, actions = lqg_policy_evaluation(
        rng_key=key,
        num_time_steps=T,
        num_trajectory_samples=num_runs,
        system_matrices=system_matrices,
        reward_fn=reward_fn,
    )

    # DEBUG: Check shapes and sample values
    print("DEBUG: Checking outputs...")
    print(f"  Rewards shape: {rewards.shape}")  # Should be (T, num_runs)
    print(f"  States shape: {states.shape}")    # Should be (T+1, num_runs, state_dim)
    print(f"  Actions shape: {actions.shape}")  # Should be (T, num_runs, action_dim)

    # Sample trajectory debugging
    traj_idx = 0
    print(f"\nDEBUG: Sample trajectory {traj_idx}:")
    print(f"  Initial state: {states[0, traj_idx, :]}")
    print(f"  Final state: {states[-1, traj_idx, :]}")
    print(f"  Sample actions: {actions[:5, traj_idx, :].flatten()}")  # First 5 actions
    print(f"  Sample rewards: {rewards[:5, traj_idx]}")  # First 5 rewards
    print(f"  Cumulative reward: {jnp.sum(rewards[:, traj_idx])}")

    # Check reward statistics
    print(f"\nDEBUG: Reward statistics:")
    print(f"  Mean reward per step: {jnp.mean(rewards):.6f}")
    print(f"  Std reward per step: {jnp.std(rewards):.6f}")
    print(f"  Min reward: {jnp.min(rewards):.6f}")
    print(f"  Max reward: {jnp.max(rewards):.6f}")

    # Compute cumulative rewards for each trajectory
    cumulative_rewards = jnp.sum(rewards, axis=0)  # Sum over time steps
    cumulative_costs = -cumulative_rewards  # Convert rewards back to costs

    # Compute statistics
    mean_cost = jnp.mean(cumulative_costs)
    std_cost = jnp.std(cumulative_costs, ddof=1)  # Sample standard deviation
    sem_cost = std_cost / jnp.sqrt(num_runs)  # Standard error of mean

    # Additional metrics from final states
    final_states = states[-1, :, :]  # Shape: (num_runs, state_dim)
    final_position_errors = jnp.abs(final_states[:, 0])
    final_velocity_errors = jnp.abs(final_states[:, 1])

    mean_pos_error = jnp.mean(final_position_errors)
    mean_vel_error = jnp.mean(final_velocity_errors)

    # Print some sample trajectories
    print(f"Sample Initial States:")
    for i in range(min(5, num_runs)):
        init_state = states[0, i, :]
        print(f"  Traj {i+1:2d}: pos={init_state[0]:6.3f}, vel={init_state[1]:6.3f}")
    print()

    # Print results
    print("BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Cumulative Cost Statistics ({num_runs} trajectories):")
    print(f"  Mean ± SEM:     {mean_cost:.3f} ± {sem_cost:.3f}")
    print(f"  Std Dev:        {std_cost:.3f}")
    print(f"  Min Cost:       {jnp.min(cumulative_costs):.3f}")
    print(f"  Max Cost:       {jnp.max(cumulative_costs):.3f}")
    print()
    print(f"Final State Errors:")
    print(f"  Mean |position|: {mean_pos_error:.6f}")
    print(f"  Mean |velocity|: {mean_vel_error:.6f}")
    print()
    print(f"Performance Summary:")
    print(f"  Average Cost:    {mean_cost:.3f}")
    print(f"  95% CI:         [{mean_cost - 1.96*sem_cost:.3f}, {mean_cost + 1.96*sem_cost:.3f}]")

    return {
        'mean_cost': mean_cost,
        'sem_cost': sem_cost,
        'std_cost': std_cost,
        'all_costs': cumulative_costs,
        'mean_pos_error': mean_pos_error,
        'mean_vel_error': mean_vel_error,
        'cumulative_rewards': cumulative_rewards,
        'final_states': final_states,
        'system_matrices': system_matrices
    }


def main():
    """Run the LQG benchmark using SMC-style evaluation."""

    # Run benchmark matching SMC policy evaluation style
    benchmark_results = run_lqg_benchmark(
        num_runs=1000,  # Match SMC trajectory samples
        T=25,           # Match Linear2D environment time steps
        seed=42
    )

    print("\n" + "="*50)
    print("FINAL BENCHMARK SUMMARY")
    print("="*50)
    print(f" Mean Cumulative Cost: {benchmark_results['mean_cost']:.3f}")
    print(f"Standard Error (SEM): {benchmark_results['sem_cost']:.3f}")
    print(f"95% Confidence Interval: [{benchmark_results['mean_cost'] - 1.96*benchmark_results['sem_cost']:.3f}, "
          f"{benchmark_results['mean_cost'] + 1.96*benchmark_results['sem_cost']:.3f}]")
    print()
    print("OPTIMAL BASELINE ESTABLISHED!")
    print("This LQG performance mirrors SMC policy_evaluation structure.")
    print("Direct comparison with SMC methods is now accurate!")


if __name__ == "__main__":
    main()