from functools import partial

import jax
from jax import Array, numpy as jnp

from distrax import (
    Deterministic,
    MultivariateNormalDiag
)

from ppomdp.core import PRNGKey, TransitionModel, ObservationModel
from ppomdp.envs.core import POMDPEnv


# Environment dimensions
state_dim = 2   # [position, velocity]
action_dim = 1  # control input
obs_dim = 1     # position only (velocity hidden)

num_envs = 1
num_time_steps = 50

# Time step for discrete-time dynamics
dt = 0.1

# System matrices for linear dynamics (double integrator)
# x_{t+1} = A @ x_t + B @ u_t + w_t
# y_t = C @ x_t + v_t

A = jnp.array([
    [1.0, dt],   # position = position + dt * velocity
    [0.0, 1.0]   # velocity = velocity + dt * action
])

B = jnp.array([
    [0.0],       # position not directly controlled
    [dt]         # velocity controlled with dt scaling
])

C = jnp.array([
    [1.0, 0.0]   # observe position only, velocity hidden
])

# Process noise covariance
Q_process = jnp.array([
    [1e-4, 0.0],
    [0.0, 1e-3]
])

# Observation noise covariance
R_obs = jnp.array([[1e-3]])

# Cost matrices for quadratic reward
Q_cost = jnp.array([
    [1.0, 0.0],      # penalize position deviation
    [0.0, 0.1]       # penalize velocity (smaller weight)
])

R_cost = jnp.array([[0.001]])  # penalize control effort


def mean_trans(s: Array, a: Array) -> Array:
    """Mean of transition dynamics: x_{t+1} = A @ x_t + B @ u_t"""
    return A @ s + B @ a


def stddev_trans(s: Array, a: Array) -> Array:
    """Standard deviation of transition noise (diagonal of Q_process)"""
    return jnp.sqrt(jnp.diag(Q_process))


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    """Sample next state from linear-Gaussian transition model"""
    dist = MultivariateNormalDiag(
        loc=mean_trans(s, a),
        scale_diag=stddev_trans(s, a)
    )
    return dist.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    """Log probability of transition s -> sn given action a"""
    dist = MultivariateNormalDiag(
        loc=mean_trans(s, a),
        scale_diag=stddev_trans(s, a)
    )
    return dist.log_prob(sn)


def mean_obs(s: Array) -> Array:
    """Mean of observation: y = C @ x (observe position only)"""
    return C @ s


def stddev_obs(s: Array) -> Array:
    """Standard deviation of observation noise"""
    return jnp.sqrt(jnp.diag(R_obs))


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    """Sample observation from linear-Gaussian observation model"""
    dist = MultivariateNormalDiag(
        loc=mean_obs(s),
        scale_diag=stddev_obs(s)
    )
    return dist.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    """Log probability of observation z given state s"""
    dist = MultivariateNormalDiag(
        loc=mean_obs(s),
        scale_diag=stddev_obs(s)
    )
    return dist.log_prob(z)


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    """Quadratic reward function: -0.5 * (x^T Q x + u^T R u)

    Negative because we want to minimize cost, but the algorithms maximize reward.
    """
    # Quadratic cost in state
    state_cost = s.T @ Q_cost @ s

    # Quadratic cost in action
    action_cost = a.T @ R_cost @ a

    # Return negative cost as reward
    return -0.5 * (state_cost + action_cost)


# Initial state distribution
init_dist = MultivariateNormalDiag(
    loc=jnp.array([1., 0.]),
    scale_diag=jnp.array([0.1, 0.1])  # small initial uncertainty
)

# Belief prior (same as initial distribution)
belief_prior = MultivariateNormalDiag(
    loc=jnp.array([1., 0.]),
    scale_diag=jnp.array([0.1, 0.1])
)

# Create transition and observation models
trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)


@partial(jnp.vectorize, signature="(n)->(m)")
def feature_fn(state: Array) -> Array:
    return state


# Create the POMDP environment
Linear2DEnv = POMDPEnv(
    num_envs=num_envs,
    state_dim=state_dim,
    action_dim=action_dim,
    obs_dim=obs_dim,
    num_time_steps=num_time_steps,
    init_dist=init_dist,
    belief_prior=belief_prior,
    trans_model=trans_model,
    obs_model=obs_model,
    reward_fn=reward_fn,
    feature_fn=feature_fn,
)


# Export system matrices for LQG comparison
LINEAR_2D_MATRICES = {
    'A': A,
    'B': B,
    'C': C,
    'Q_process': Q_process,
    'R_obs': R_obs,
    'Q_cost': Q_cost,
    'R_cost': R_cost,
    'dt': dt,
    'init_loc': jnp.array([1., 0.]),     # initial location from init_dist
    'init_scale_diag': jnp.array([0.1, 0.1])  # initial uncertainty from init_dist
}