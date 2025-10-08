"""Linear-Quadratic-Gaussian (LQG) controller.

Implements optimal control for linear-Gaussian systems with quadratic costs
using the separation principle: LQR control applied to Kalman filter estimates.
"""

from typing import NamedTuple, Tuple, Callable
import jax
from jax import Array, numpy as jnp
import jax.random as random
from jax.random import PRNGKey

from .lqr import LQRController, solve_lqr, lqr_control
from .kalman import KalmanFilter, BeliefState


class LQGPolicy:
    """LQG policy combining LQR control with Kalman filtering.

    Implements the separation principle for optimal control under partial observability:
    1. Use Kalman filter to estimate state from observations
    2. Apply LQR control to the state estimate
    """

    def __init__(
        self,
        lqr_controller: LQRController,
        kalman_filter: KalmanFilter,
        initial_belief: BeliefState
    ):
        """Initialize LQG policy.

        Args:
            lqr_controller: Optimal LQR controller
            kalman_filter: Kalman filter for state estimation
            initial_belief: Initial belief state
        """
        self.lqr = lqr_controller
        self.kf = kalman_filter
        self.initial_belief = initial_belief
        self.belief = initial_belief

        # Validate compatibility
        assert self.lqr.K.shape[1] == self.kf.state_dim, \
            "LQR and Kalman filter state dimensions must match"

    def reset(self, initial_belief: BeliefState = None) -> None:
        """Reset policy to initial belief state.

        Args:
            initial_belief: Optional new initial belief. If None, uses default.
        """
        if initial_belief is not None:
            self.belief = initial_belief
        else:
            self.belief = self.initial_belief

    def get_action(self, observation: Array = None) -> Array:
        """Get control action for current belief state.

        Args:
            observation: Optional observation to update belief first

        Returns:
            Control action based on current belief mean
        """
        if observation is not None:
            # Update belief with observation
            self.belief = self.kf.update(self.belief, observation)

        # Apply LQR control to belief mean
        action = lqr_control(self.belief.mean, self.lqr.K)
        return action

    def step(self, action: Array, observation: Array) -> Array:
        """Full step: predict with action, update with observation, get next action.

        Args:
            action: Previous action taken
            observation: Current observation

        Returns:
            Next control action
        """
        # Update belief: predict + update
        self.belief = self.kf.step(self.belief, action, observation)

        # Get next action based on updated belief
        next_action = lqr_control(self.belief.mean, self.lqr.K)
        return next_action

    @property
    def state_estimate(self) -> Array:
        """Current state estimate (belief mean)."""
        return self.belief.mean

    def state_uncertainty(self) -> Array:
        """Current state uncertainty (belief covariance)."""
        return self.belief.covar


def create_lqg_policy(
    A: Array,
    B: Array,
    C: Array,
    Q_dyn: Array,  # Process noise covariance
    R_obs: Array,  # Observation noise covariance
    Q_cost: Array, # State cost matrix
    R_cost: Array, # Control cost matrix
    initial_mean: Array,
    initial_cov: Array
) -> LQGPolicy:
    """Create LQG policy from system matrices.

    Args:
        A: State transition matrix (state_dim, state_dim)
        B: Control input matrix (state_dim, action_dim)
        C: Observation matrix (obs_dim, state_dim)
        Q_dyn: Process noise covariance (state_dim, state_dim)
        R_obs: Observation noise covariance (obs_dim, obs_dim)
        Q_cost: State cost matrix for LQR (state_dim, state_dim)
        R_cost: Control cost matrix for LQR (action_dim, action_dim)
        initial_mean: Initial state mean (state_dim,)
        initial_cov: Initial state covariance (state_dim, state_dim)

    Returns:
        LQG policy ready for control
    """
    # Solve LQR problem
    lqr_controller = solve_lqr(A, B, Q_cost, R_cost)

    # Create Kalman filter
    kalman_filter = KalmanFilter(A, B, C, Q_dyn, R_obs)

    # Initialize belief
    initial_belief = kalman_filter.init_belief(initial_mean, initial_cov)

    # Create LQG policy
    lqg_policy = LQGPolicy(lqr_controller, kalman_filter, initial_belief)

    return lqg_policy


def lqg_simulate(
    lqg_policy: LQGPolicy,
    true_states: Array,
    observations: Array
) -> Tuple[Array, Array, Array]:
    """Simulate LQG policy over trajectory.

    Args:
        lqg_policy: LQG policy to simulate
        true_states: True state trajectory (T+1, state_dim)
        observations: Observation trajectory (T, obs_dim)

    Returns:
        actions: Control actions (T, action_dim)
        state_estimates: State estimates (T+1, state_dim)
        costs: Control costs (T,)
    """
    T = observations.shape[0]
    action_dim = lqg_policy.lqr.K.shape[0]
    state_dim = lqg_policy.kf.state_dim

    # Initialize storage
    actions = jnp.zeros((T, action_dim))
    state_estimates = jnp.zeros((T + 1, state_dim))
    costs = jnp.zeros(T)

    # Reset policy
    belief = lqg_policy.initial_belief
    state_estimates = state_estimates.at[0].set(belief.mean)

    # Simulate trajectory
    for t in range(T):
        # Get control action
        action = lqr_control(belief.mean, lqg_policy.lqr.K)
        actions = actions.at[t].set(action)

        # Compute cost (using true state for evaluation)
        state_cost = true_states[t].T @ lqg_policy.lqr.P @ true_states[t]
        control_cost = action.T @ action  # Assuming R_cost = I for simplicity
        costs = costs.at[t].set(state_cost + control_cost)

        # Update belief with observation
        predicted = lqg_policy.kf.predict(belief, action)
        belief = lqg_policy.kf.update(predicted, observations[t])
        state_estimates = state_estimates.at[t + 1].set(belief.mean)

    return actions, state_estimates, costs


class LQGConfig(NamedTuple):
    """Configuration for LQG system."""
    # System matrices
    A: Array  # State transition
    B: Array  # Control input
    C: Array  # Observation

    # Noise covariances
    Q_dyn: Array   # Process noise
    R_obs: Array   # Observation noise

    # Cost matrices
    Q_cost: Array  # State cost
    R_cost: Array  # Control cost

    # Initial conditions
    initial_mean: Array  # Initial state mean
    initial_cov: Array   # Initial state covariance


def evaluate_lqg_performance(
    config: LQGConfig,
    true_trajectory: Array,
    observation_trajectory: Array
) -> dict:
    """Evaluate LQG performance on a given trajectory.

    Args:
        config: LQG system configuration
        true_trajectory: True state trajectory (T+1, state_dim)
        observation_trajectory: Observations (T, obs_dim)

    Returns:
        Dictionary with performance metrics
    """
    # Create LQG policy
    lqg_policy = create_lqg_policy(
        config.A, config.B, config.C,
        config.Q_dyn, config.R_obs,
        config.Q_cost, config.R_cost,
        config.initial_mean, config.initial_cov
    )

    # Simulate policy
    actions, estimates, costs = lqg_simulate(
        lqg_policy, true_trajectory, observation_trajectory
    )

    # Compute metrics
    total_cost = jnp.sum(costs)
    mean_cost = jnp.mean(costs)
    estimation_error = jnp.mean(jnp.linalg.norm(
        estimates[1:] - true_trajectory[1:], axis=1
    ))

    return {
        'total_cost': total_cost,
        'mean_cost': mean_cost,
        'estimation_error': estimation_error,
        'actions': actions,
        'state_estimates': estimates,
        'costs': costs
    }


def lqg_policy_evaluation(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_trajectory_samples: int,
    system_matrices: tuple,
    reward_fn: Callable,
) -> tuple[Array, Array, Array]:
    """Evaluates an LQG policy following the same style as SMC policy_evaluation.

    This function mirrors the exact structure of the SMC policy evaluation to ensure
    fair comparison between LQG and SMC methods.

    Args:
        rng_key: Random number generator key
        num_time_steps: Number of time steps to evaluate
        num_trajectory_samples: Number of sampled trajectories
        system_matrices: Tuple of (A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_cov)
        reward_fn: Reward function of the POMDP environment

    Returns:
        Tuple containing:
            - rewards: Array of rewards for each time step and trajectory
            - states: Array of states for each time step and trajectory
            - actions: Array of actions for each time step and trajectory
    """
    A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_cov = system_matrices

    # Pre-compute LQR controller (same for all trajectories)
    lqr_controller = solve_lqr(A, B, Q_cost, R_cost)

    def run_single_trajectory(traj_key, initial_state):
        """Run a single LQG trajectory matching SMC style exactly."""

        # Create Kalman filter with fresh belief for this trajectory
        kalman_filter = KalmanFilter(A, B, C, Q_dyn, R_obs)
        belief = BeliefState(mean=initial_state, covar=initial_cov)

        def body(val, key):
            state, belief, time_idx = val

            # Sample observation from current state
            key, obs_key = random.split(key)
            obs_noise = random.multivariate_normal(obs_key, jnp.zeros(1), R_obs)
            observation = C @ state + obs_noise

            # Update belief with observation (Kalman filter update step)
            updated_belief = kalman_filter.update(belief, observation)

            # Get action from LQR controller using updated belief mean
            action = lqr_control(updated_belief.mean, lqr_controller.K)

            # Sample next state (matching SMC style)
            key, state_key = random.split(key)
            process_noise = random.multivariate_normal(state_key, jnp.zeros(2), Q_dyn)
            next_state = A @ state + B @ action.reshape(-1) + process_noise

            # Compute reward using NEW state and action (matching SMC style)
            reward = reward_fn(next_state, action, time_idx)

            # Predict next belief for next iteration (Kalman filter predict step)
            next_belief = kalman_filter.predict(updated_belief, action)

            return (next_state, next_belief, time_idx + 1), (next_state, action, reward)

        # Initial belief matches the true initial state
        init_belief = BeliefState(mean=initial_state, covar=initial_cov)

        # Run trajectory scan (matching SMC structure exactly)
        _, (states, actions, rewards) = jax.lax.scan(
            f=body,
            init=(initial_state, init_belief, 1),  # time_idx starts at 1 like SMC
            xs=random.split(traj_key, num_time_steps),
        )

        return states, actions, rewards

    # Sample initial states (using Linear2D environment's exact distribution)
    key, state_key = random.split(rng_key)
    from ppomdp.envs.pomdps.linear2d import init_dist
    init_states = init_dist.sample(seed=state_key, sample_shape=num_trajectory_samples)

    # Generate keys for each trajectory
    traj_keys = random.split(key, num_trajectory_samples)

    # Run multiple trajectories (each with independent belief state)
    all_states, all_actions, all_rewards = \
        jax.vmap(run_single_trajectory, in_axes=(0, 0))(traj_keys, init_states)

    # Transpose arrays to match SMC style: (time, trajectories, dim)
    rewards = all_rewards.T  # (num_time_steps, num_trajectory_samples)
    actions = all_actions.transpose((1, 0, 2))  # (num_time_steps, num_trajectory_samples, action_dim)
    states = all_states.transpose((1, 0, 2))  # (num_time_steps, num_trajectory_samples, state_dim)

    # Concatenate initial states (matching SMC style)
    init_states_expanded = init_states[None, :, :]  # (1, num_trajectory_samples, state_dim)
    states = jnp.concatenate([init_states_expanded, states], axis=0)  # (num_time_steps+1, num_trajectory_samples, state_dim)

    return rewards, states, actions
