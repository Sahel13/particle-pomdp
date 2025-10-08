"""LQG (Linear-Quadratic-Gaussian) solver package.

This package implements optimal control for linear-Gaussian systems with quadratic costs.
The solution follows the separation principle: LQR control + Kalman filtering.
"""

from .lqr import solve_lqr, LQRController
from .kalman import KalmanFilter, BeliefState
from .lqg import LQGPolicy, create_lqg_policy, LQGConfig, evaluate_lqg_performance, lqg_policy_evaluation

__all__ = [
    'solve_lqr',
    'LQRController', 
    'KalmanFilter',
    'BeliefState',
    'LQGPolicy',
    'create_lqg_policy',
    'LQGConfig',
    'evaluate_lqg_performance',
    'lqg_policy_evaluation'
] 