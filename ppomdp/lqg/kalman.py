"""Kalman Filter for state estimation in linear-Gaussian systems.

Implements optimal state estimation for systems:
- Dynamics: x_{t+1} = A*x_t + B*u_t + w_t, w_t ~ N(0, Q)  
- Observations: y_t = C*x_t + v_t, v_t ~ N(0, R)
"""

from typing import NamedTuple, Tuple
import jax
from jax import Array, numpy as jnp
from jax.scipy.linalg import solve


class BeliefState(NamedTuple):
    """Gaussian belief state with mean and covariance."""
    mean: Array      # State mean (state_dim,)
    covar: Array  # State covariance (state_dim, state_dim)


class KalmanFilter:
    """Kalman filter for linear-Gaussian state estimation.
    
    Maintains Gaussian belief over state and updates with observations.
    """
    
    def __init__(
        self,
        A: Array,  # State transition (state_dim, state_dim)
        B: Array,  # Control input (state_dim, action_dim)  
        C: Array,  # Observation (obs_dim, state_dim)
        Q: Array,  # Process noise covariance (state_dim, state_dim)
        R: Array,  # Observation noise covariance (obs_dim, obs_dim)
    ):
        """Initialize Kalman filter with system matrices.
        
        Args:
            A: State transition matrix
            B: Control input matrix
            C: Observation matrix  
            Q: Process noise covariance (must be PSD)
            R: Observation noise covariance (must be PD)
        """
        self.A = A
        self.B = B  
        self.C = C
        self.Q = Q
        self.R = R
        
        # Validate dimensions
        self.state_dim, self.action_dim = B.shape
        self.obs_dim, state_dim2 = C.shape
        assert state_dim2 == self.state_dim, "Inconsistent state dimensions"
        assert A.shape == (self.state_dim, self.state_dim), "A matrix wrong shape"
        assert Q.shape == (self.state_dim, self.state_dim), "Q matrix wrong shape"  
        assert R.shape == (self.obs_dim, self.obs_dim), "R matrix wrong shape"
    
    def init_belief(self, mean: Array, covar: Array) -> BeliefState:
        """Initialize belief state.
        
        Args:
            mean: Initial state mean (state_dim,)
            covariance: Initial state covariance (state_dim, state_dim)
            
        Returns:
            Initial belief state
        """
        assert mean.shape == (self.state_dim,), f"Mean must be shape ({self.state_dim},)"
        assert covar.shape == (self.state_dim, self.state_dim), \
            f"Covariance must be shape ({self.state_dim}, {self.state_dim})"
        return BeliefState(mean=mean, covar=covar)
    
    def predict(self, belief: BeliefState, action: Array) -> BeliefState:
        """Prediction step: propagate belief through dynamics.
        
        Args:
            belief: Current belief state
            action: Control action (action_dim,)
            
        Returns:
            Predicted belief state
        """
        # Predict mean: mu_{t+1|t} = A*mu_t + B*u_t
        pred_mean = self.A @ belief.mean + self.B @ action
        
        # Predict covariance: Sigma_{t+1|t} = A*Sigma_t*A^T + Q
        pred_cov = self.A @ belief.covar @ self.A.T + self.Q
        
        return BeliefState(mean=pred_mean, covar=pred_cov)
    
    def update(self, belief: BeliefState, observation: Array) -> BeliefState:
        """Update step: incorporate observation into belief.
        
        Args:
            belief: Predicted belief state
            observation: Observation (obs_dim,)
            
        Returns:
            Updated belief state
        """
        # Innovation: y_t - C*mu_{t|t-1}
        innovation = observation - self.C @ belief.mean
        
        # Innovation covariance: S = C*Sigma_{t|t-1}*C^T + R
        innovation_cov = self.C @ belief.covar @ self.C.T + self.R
        
        # Kalman gain: K = Sigma_{t|t-1}*C^T*S^{-1}
        kalman_gain = belief.covar @ self.C.T @ jnp.linalg.inv(innovation_cov)
        
        # Updated mean: mu_{t|t} = mu_{t|t-1} + K*innovation
        updated_mean = belief.mean + kalman_gain @ innovation
        
        # Updated covariance: Sigma_{t|t} = (I - K*C)*Sigma_{t|t-1}
        I_KC = jnp.eye(self.state_dim) - kalman_gain @ self.C
        updated_cov = I_KC @ belief.covar
        
        return BeliefState(mean=updated_mean, covar=updated_cov)
    
    def step(self, belief: BeliefState, action: Array, observation: Array) -> BeliefState:
        """Combined predict + update step.
        
        Args:
            belief: Current belief state  
            action: Control action
            observation: Observation
            
        Returns:
            Updated belief state after prediction and observation
        """
        predicted = self.predict(belief, action)
        updated = self.update(predicted, observation)
        return updated
    
    def log_likelihood(self, belief: BeliefState, observation: Array) -> Array:
        """Compute log-likelihood of observation given predicted belief.
        
        Args:
            belief: Predicted belief state
            observation: Observation
            
        Returns:
            Log-likelihood of observation
        """
        # Innovation and covariance
        innovation = observation - self.C @ belief.mean
        innovation_cov = self.C @ belief.covar @ self.C.T + self.R
        
        # Multivariate normal log-likelihood
        sign, log_det = jnp.linalg.slogdet(innovation_cov)
        quad_form = innovation.T @ jnp.linalg.inv(innovation_cov) @ innovation
        
        log_likelihood = -0.5 * (
            self.obs_dim * jnp.log(2 * jnp.pi) + 
            log_det + 
            quad_form
        )
        
        return log_likelihood


@jax.jit
def batch_kalman_filter(
    kf: KalmanFilter,
    initial_belief: BeliefState, 
    actions: Array,
    observations: Array
) -> Tuple[Array, Array, Array]:
    """Run Kalman filter over a sequence of actions and observations.
    
    Args:
        kf: Kalman filter instance
        initial_belief: Initial belief state
        actions: Action sequence (T, action_dim)
        observations: Observation sequence (T, obs_dim)
        
    Returns:
        means: State estimates (T+1, state_dim) 
        covariances: State covariances (T+1, state_dim, state_dim)
        log_likelihoods: Observation log-likelihoods (T,)
    """
    T = actions.shape[0]
    state_dim = initial_belief.mean.shape[0]
    
    # Initialize storage
    means = jnp.zeros((T + 1, state_dim))
    covariances = jnp.zeros((T + 1, state_dim, state_dim))
    log_likelihoods = jnp.zeros(T)
    
    # Set initial belief
    means = means.at[0].set(initial_belief.mean)
    covariances = covariances.at[0].set(initial_belief.covar)
    
    belief = initial_belief
    
    # Run filter
    for t in range(T):
        # Predict
        predicted = kf.predict(belief, actions[t])
        
        # Log-likelihood of observation
        log_lik = kf.log_likelihood(predicted, observations[t])
        log_likelihoods = log_likelihoods.at[t].set(log_lik)
        
        # Update  
        belief = kf.update(predicted, observations[t])
        
        # Store results
        means = means.at[t + 1].set(belief.mean)
        covariances = covariances.at[t + 1].set(belief.covar)
    
    return means, covariances, log_likelihoods 