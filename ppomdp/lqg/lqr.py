"""Linear-Quadratic Regulator (LQR) solver.

Implements optimal control for linear systems with quadratic costs:
- Dynamics: x_{t+1} = A*x_t + B*u_t + w_t
- Cost: sum_t (x_t^T*Q*x_t + u_t^T*R*u_t)
- Solution: u_t = -K*x_t where K is optimal gain matrix
"""

from typing import NamedTuple, Tuple
import jax
from jax import Array, numpy as jnp
from jax.scipy.linalg import solve


class LQRController(NamedTuple):
    """LQR controller containing optimal gain matrix and cost-to-go."""
    K: Array  # Optimal gain matrix (action_dim, state_dim)
    P: Array  # Cost-to-go matrix (state_dim, state_dim)


def solve_lqr(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    max_iters: int = 1000,
    tolerance: float = 1e-8
) -> LQRController:
    """Solve discrete-time LQR problem using iterative Riccati equation.

    Solves for optimal control law u = -K*x that minimizes:
    J = sum_t (x_t^T*Q*x_t + u_t^T*R*u_t)

    Subject to: x_{t+1} = A*x_t + B*u_t

    Args:
        A: State transition matrix (state_dim, state_dim)
        B: Control input matrix (state_dim, action_dim)
        Q: State cost matrix (state_dim, state_dim), must be PSD
        R: Control cost matrix (action_dim, action_dim), must be PD
        max_iters: Maximum iterations for convergence
        tolerance: Convergence tolerance

    Returns:
        LQRController with optimal gain K and cost-to-go P
    """
    # Validate inputs
    state_dim, action_dim = B.shape
    assert A.shape == (state_dim, state_dim), f"A must be {state_dim}x{state_dim}"
    assert Q.shape == (state_dim, state_dim), f"Q must be {state_dim}x{state_dim}"
    assert R.shape == (action_dim, action_dim), f"R must be {action_dim}x{action_dim}"

    # Initialize P with Q
    P = Q.copy()

    for i in range(max_iters):
        P_prev = P.copy()

        # Solve for optimal gain: K = (B^T*P*B + R)^{-1} * B^T*P*A
        BtPB_R = B.T @ P @ B + R
        BtPA = B.T @ P @ A
        K = solve(BtPB_R, BtPA, assume_a='pos')

        # Update P: P = A^T*P*A - K^T*(B^T*P*B + R)*K + Q
        AtPA = A.T @ P @ A
        KtRK = K.T @ (B.T @ P @ B + R) @ K
        P = AtPA - KtRK + Q

        # Check convergence
        error = jnp.max(jnp.abs(P - P_prev))
        if error < tolerance:
            break
    else:
        print(f"Warning: LQR did not converge in {max_iters} iterations (error={error:.2e})")

    return LQRController(K=K, P=P)


def solve_lqr_finite_horizon(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    Qf: Array,
    T: int
) -> Tuple[Array, Array]:
    """Solve finite-horizon LQR problem.

    Args:
        A, B, Q, R: System matrices
        Qf: Terminal cost matrix
        T: Time horizon

    Returns:
        K: Gain matrices (T, action_dim, state_dim)
        P: Cost-to-go matrices (T+1, state_dim, state_dim)
    """
    state_dim, action_dim = B.shape

    # Initialize arrays
    K = jnp.zeros((T, action_dim, state_dim))
    P = jnp.zeros((T + 1, state_dim, state_dim))
    P = P.at[-1].set(Qf)  # Terminal condition

    # Backward recursion
    for t in range(T - 1, -1, -1):
        # Optimal gain at time t
        BtPB_R = B.T @ P[t + 1] @ B + R
        BtPA = B.T @ P[t + 1] @ A
        K_t = solve(BtPB_R, BtPA, assume_a='pos')
        K = K.at[t].set(K_t)

        # Cost-to-go at time t
        AtPA = A.T @ P[t + 1] @ A
        KtRK = K_t.T @ (B.T @ P[t + 1] @ B + R) @ K_t
        P = P.at[t].set(AtPA - KtRK + Q)

    return K, P


@jax.jit
def lqr_control(state: Array, K: Array) -> Array:
    """Apply LQR control law: u = -K @ x

    Args:
        state: Current state (state_dim,) or (..., state_dim)
        K: LQR gain matrix (action_dim, state_dim)

    Returns:
        control: Optimal control action (action_dim,) or (..., action_dim)
    """
    return -K @ state
