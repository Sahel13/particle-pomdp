#!/usr/bin/env python3
"""Tests for LQG solver components."""

import jax
jax.config.update("jax_enable_x64", True)

import pytest
from jax import random, numpy as jnp

from ppomdp.lqg import solve_lqr, KalmanFilter, create_lqg_policy


def test_lqr_solver():
    """Test LQR solver on simple 1D system."""
    # Simple 1D double integrator: x_{t+1} = x_t + dt*v_t, v_{t+1} = v_t + dt*u_t
    dt = 0.1
    A = jnp.array([[1.0, dt], [0.0, 1.0]])
    B = jnp.array([[0.0], [dt]])
    Q = jnp.array([[1.0, 0.0], [0.0, 0.1]])  # penalize position more
    R = jnp.array([[0.01]])  # small control penalty

    # Solve LQR
    lqr = solve_lqr(A, B, Q, R)

    # Check dimensions
    assert lqr.K.shape == (1, 2), f"Gain matrix should be (1,2), got {lqr.K.shape}"
    assert lqr.P.shape == (2, 2), f"Cost-to-go should be (2,2), got {lqr.P.shape}"

    # Check that P is positive semidefinite
    eigenvals = jnp.linalg.eigvals(lqr.P)
    assert jnp.all(eigenvals >= -1e-10), f"P should be PSD, eigenvals: {eigenvals}"

    # Check stability: closed-loop system should be stable
    A_cl = A - B @ lqr.K
    eigenvals_cl = jnp.linalg.eigvals(A_cl)
    assert jnp.all(jnp.abs(eigenvals_cl) < 1.0), f"Closed-loop should be stable, eigenvals: {eigenvals_cl}"

    print(f"✓ LQR test passed. Gain: {lqr.K.flatten()}")


def test_kalman_filter():
    """Test Kalman filter on simple system."""
    # Same 1D double integrator system
    dt = 0.1
    A = jnp.array([[1.0, dt], [0.0, 1.0]])
    B = jnp.array([[0.0], [dt]])
    C = jnp.array([[1.0, 0.0]])  # observe position only
    Q = 1e-4 * jnp.eye(2)  # process noise
    R = jnp.array([[1e-3]])  # observation noise

    # Create Kalman filter
    kf = KalmanFilter(A, B, C, Q, R)

    # Test initialization
    initial_mean = jnp.array([1.0, 0.0])
    initial_cov = 0.1 * jnp.eye(2)
    belief = kf.init_belief(initial_mean, initial_cov)

    assert belief.mean.shape == (2,), f"Belief mean should be (2,), got {belief.mean.shape}"
    assert belief.covar.shape == (2, 2), f"Belief cov should be (2,2), got {belief.covar.shape}"

    # Test prediction step
    action = jnp.array([0.1])
    predicted = kf.predict(belief, action)

    # Mean should follow dynamics: x_{t+1} = A*x_t + B*u_t
    expected_mean = A @ belief.mean + B @ action
    assert jnp.allclose(predicted.mean, expected_mean), "Prediction mean incorrect"

    # Test update step
    observation = jnp.array([1.1])  # noisy observation of position
    updated = kf.update(predicted, observation)

    # Updated belief should have smaller uncertainty (for observed state)
    assert updated.covar[0, 0] < predicted.covar[0, 0], "Update should reduce position uncertainty"

    print(f" Kalman filter test passed.")


def test_lqg_policy():
    """Test full LQG policy integration."""
    # System matrices
    dt = 0.1
    A = jnp.array([[1.0, dt], [0.0, 1.0]])
    B = jnp.array([[0.0], [dt]])
    C = jnp.array([[1.0, 0.0]])

    # Noise covariances
    Q_dyn = 1e-4 * jnp.eye(2)
    R_obs = jnp.array([[1e-3]])

    # Cost matrices
    Q_cost = jnp.array([[1.0, 0.0], [0.0, 0.1]])
    R_cost = jnp.array([[0.01]])

    # Initial conditions
    initial_mean = jnp.array([2.0, 0.5])
    initial_cov = 0.1 * jnp.eye(2)

    # Create LQG policy
    lqg_policy = create_lqg_policy(
        A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_mean, initial_cov
    )

    # Test action generation
    observation = jnp.array([1.8])  # position observation
    action = lqg_policy.get_action(observation)

    assert action.shape == (1,), f"Action should be (1,), got {action.shape}"
    assert jnp.isfinite(action).all(), "Action should be finite"

    # Test that control reduces position (since we're trying to stabilize to origin)
    # For positive position, expect negative control (to move toward origin)
    if observation[0] > 0:
        assert action[0] < 0, f"For positive position {observation[0]}, expected negative action, got {action[0]}"

    # Test step function
    next_action = lqg_policy.step(action, observation)
    assert next_action.shape == (1,), f"Next action should be (1,), got {next_action.shape}"

    print(f"  LQG policy test passed. Action for obs {observation[0]}: {action[0]:.3f}")


def test_lqg_stability():
    """Test that LQG policy stabilizes the system."""
    # Create simple system
    dt = 0.1
    A = jnp.array([[1.0, dt], [0.0, 1.0]])
    B = jnp.array([[0.0], [dt]])
    C = jnp.array([[1.0, 0.0]])
    Q_dyn = 1e-6 * jnp.eye(2)  # very small noise
    R_obs = jnp.array([[1e-6]])
    Q_cost = jnp.array([[1.0, 0.0], [0.0, 0.1]])
    R_cost = jnp.array([[0.01]])

    initial_mean = jnp.array([5.0, 1.0])  # start far from origin
    initial_cov = 1e-6 * jnp.eye(2)  # very certain

    lqg_policy = create_lqg_policy(
        A, B, C, Q_dyn, R_obs, Q_cost, R_cost, initial_mean, initial_cov
    )

    # Simulate (deterministic, no noise)
    state = initial_mean.copy()
    lqg_policy.reset()

    positions = []
    for t in range(100):
        # Perfect observation (no noise)
        observation = jnp.array([state[0]])
        action = lqg_policy.get_action(observation)

        # Perfect dynamics (no noise)
        state = A @ state + B @ action.flatten()
        positions.append(state[0])

        # Update LQG belief
        lqg_policy.belief = lqg_policy.kf.predict(lqg_policy.belief, action)

    # Check convergence to origin
    final_position = abs(positions[-1])
    initial_position = abs(positions[0])

    assert final_position < 0.1, f"System should stabilize near origin, final pos: {final_position}"
    assert final_position < initial_position, f"System should converge: {initial_position} -> {final_position}"

    print(f"  LQG stability test passed. Converged from {initial_position:.3f} to {final_position:.6f}")


def run_all_tests():
    """Run all LQG tests."""
    print("Running LQG Tests")
    print("=" * 30)

    test_lqr_solver()
    test_kalman_filter()
    test_lqg_policy()
    test_lqg_stability()

    print("\n✅ All LQG tests passed!")


if __name__ == "__main__":
    run_all_tests()