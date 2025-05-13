"""
JAX implementation of the ball catching model.
This module provides pure functions for:
1. Transition model (sample and log pdf)
2. Observation model (sample and log pdf)
3. Reward function
"""

import jax.numpy as jnp
from distrax import Block, Deterministic, MultivariateNormalDiag, ScalarAffine
from jax import Array, lax

from ppomdp.core import ObservationModel, PRNGKey, TransitionModel
from ppomdp.envs.core import POMDPEnv

# Physical constants
G = 9.81  # Gravitational constant on the surface of the Earth
MU = 10.0 / 12.0  # Air resistance per unit mass (max force 10, max velocity 12)

# Default time step
DT = 0.1
N_RK = 10  # Number of Runge-Kutta steps

# Default noise parameters
DEFAULT_SYSTEM_NOISE_SCALE = 1e-3
DEFAULT_OBS_NOISE_MIN = 1e-2  # when looking directly at the ball
DEFAULT_OBS_NOISE_MAX = 1.0  # when the ball is 90 degrees from the gaze direction
PSI_MAX = 0.8 * jnp.pi / 2

# Set action limits.
F_C1 = 7.5
F_C2 = 2.5
W_MAX = 2 * jnp.pi
PSI_MAX = 0.8 * jnp.pi / 2
action_shift = jnp.zeros(4)
action_scale = jnp.array([1.0, W_MAX, W_MAX, jnp.pi])
action_trans = Block(ScalarAffine(scale=action_scale, shift=action_shift), ndims=1)


def estimate_simulation_duration(state):
    """Estimate how long until the ball hits the ground."""
    # Extract z-coordinate and z-velocity of the ball
    _, _, z_b, _, _, vz_b, _, _, _, _, _, _ = state

    # Use kinematic equation to find time
    discriminant = vz_b**2 + 2 * G * z_b
    safe_discriminant = jnp.maximum(0.0, discriminant)  # Avoid negative inside sqrt
    T = (vz_b + jnp.sqrt(safe_discriminant)) / G

    # Divide time by time-step duration and convert to int
    return jnp.floor(T / DT).astype(jnp.int32)


# Initial condition
x_b0 = y_b0 = z_b0 = 0.0
vx_b0, vy_b0, vz_b0 = 10.0, 4.0, 15.0

x_c0, y_c0 = 22.0, 7.0
vx_c0 = vy_c0 = 0.0

phi0 = jnp.arctan2(y_b0 - y_c0, x_b0 - x_c0)  # direction towards the ball
phi0 = phi0 + 2 * jnp.pi if phi0 < 0 else phi0
psi0 = 0.0

m0 = jnp.array(
    [x_b0, y_b0, z_b0, vx_b0, vy_b0, vz_b0, x_c0, y_c0, vx_c0, vy_c0, phi0, psi0]
)

S0_diag = (
    jnp.array([0.2, 0.2, 0.0, 0.5, 0.5, 0.0, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) * 0.25
)

init_dist = Deterministic(loc=m0)
belief_prior = MultivariateNormalDiag(loc=m0, scale_diag=jnp.sqrt(S0_diag))
num_time_steps = estimate_simulation_duration(m0)


def ode(state: Array, action: Array) -> Array:
    """The ODE for the ball-catching model."""
    x_b, y_b, z_b, vx_b, vy_b, vz_b, x_c, y_c, vx_c, vy_c, phi, psi = state
    F_c, w_phi, w_psi, theta = action

    # Apply action limits.
    F_c = 0.5 * (F_c + 1) * (F_C1 + F_C2 * jnp.cos(theta))

    x_b_dot = vx_b
    y_b_dot = vy_b
    z_b_dot = vz_b
    vx_b_dot = 0.0
    vy_b_dot = 0.0
    vz_b_dot = -G
    x_c_dot = vx_c
    y_c_dot = vy_c
    vx_c_dot = F_c * jnp.cos(phi + theta) - MU * vx_c
    vy_c_dot = F_c * jnp.sin(phi + theta) - MU * vy_c
    phi_dot = w_phi
    psi_dot = w_psi

    return jnp.array(
        [
            x_b_dot,
            y_b_dot,
            z_b_dot,
            vx_b_dot,
            vy_b_dot,
            vz_b_dot,
            x_c_dot,
            y_c_dot,
            vx_c_dot,
            vy_c_dot,
            phi_dot,
            psi_dot,
        ]
    )


def rk4_step(state, action, dt):
    k1 = ode(state, action)
    k2 = ode(state + dt * k1 / 2, action)
    k3 = ode(state + dt * k2 / 2, action)
    k4 = ode(state + dt * k3, action)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def mean_trans(state, action, dt=DT, n_rk=N_RK):
    dt_rk = dt / n_rk

    def body_fn(carry, _):
        next_carry = rk4_step(carry, action, dt_rk)
        return next_carry, None

    return lax.scan(body_fn, state, length=n_rk)[0]


def stddev_trans(s: Array) -> Array:
    nx = len(s)
    M = jnp.eye(nx) * DEFAULT_SYSTEM_NOISE_SCALE

    # Catcher's dynamics is less noisy
    M = M.at[-6:, -6:].set(jnp.eye(6) * 1e-5)

    return jnp.sqrt(M)


def sample_trans(rng_key: PRNGKey, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_trans(s, a), scale_diag=stddev_trans(s))
    return dist.sample(seed=rng_key)


def log_prob_trans(sn: Array, s: Array, a: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_trans(s, a), scale_diag=stddev_trans(s))
    return dist.log_prob(sn)


def mean_obs(s: Array) -> Array:
    x_b, y_b, z_b, _, _, _, x_c, y_c, _, _, phi, psi = s
    return jnp.array([x_b, y_b, z_b, x_c, y_c, phi, psi])


def stddev_obs(state, N_min=DEFAULT_OBS_NOISE_MIN, N_max=DEFAULT_OBS_NOISE_MAX):
    x_b, y_b, z_b, _, _, _, x_c, y_c, _, _, phi, psi = state

    # Direction vector of gaze
    d = jnp.array(
        [jnp.cos(psi) * jnp.cos(phi), jnp.cos(psi) * jnp.sin(phi), jnp.sin(psi)]
    )

    # Vector from catcher to ball
    r = jnp.array([x_b - x_c, y_b - y_c, z_b])

    # Angle between gaze and ball direction
    r_norm = jnp.linalg.norm(r) + 1e-6
    cos_omega = jnp.dot(d, r) / r_norm

    # State-dependent variance
    variance = r_norm * (N_max * (1.0 - cos_omega) + N_min)

    # Create covariance matrix with zeros except for ball position observations
    N = jnp.eye(7) * 1e-5
    N = N.at[0, 0].set(variance)  # x_b variance
    N = N.at[1, 1].set(variance)  # y_b variance
    N = N.at[2, 2].set(variance)  # z_b variance

    return jnp.sqrt(N)


def sample_obs(rng_key: PRNGKey, s: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_obs(s), scale_diag=stddev_obs(s))
    return dist.sample(seed=rng_key)


def log_prob_obs(z: Array, s: Array) -> Array:
    dist = MultivariateNormalDiag(loc=mean_obs(s), scale_diag=stddev_obs(s))
    return dist.log_prob(z)


def reward_fn(s: Array, a: Array, t: Array) -> Array:
    w_cl = 1e3
    x_b, y_b, _, _, _, _, x_c, y_c, _, _, _, _ = s

    dx_bc = jnp.array([x_b - x_c, y_b - y_c])
    distance_cost = 0.5 * jnp.dot(dx_bc, dx_bc)

    R = jnp.diag(jnp.array([1e1, 1e0, 1e0, 1e-1]))

    cost = lax.select(t < num_time_steps, 0.0, w_cl * distance_cost)
    cost += 0.5 * jnp.dot(a, jnp.dot(R, a)) * DT
    return -cost


trans_model = TransitionModel(sample=sample_trans, log_prob=log_prob_trans)
obs_model = ObservationModel(sample=sample_obs, log_prob=log_prob_obs)

num_envs = 1
state_dim = 12
obs_dim = 7
action_dim = 4

LightDark2DEnv = POMDPEnv(
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
    feature_fn=lambda x: x,
)
