from typing import NamedTuple

import jax
from jax import Array, numpy as jnp, random
from distrax import Chain, MultivariateNormalDiag, Transformed
from flax import struct

from ppomdp.core import PRNGKey, Parameters, BeliefState
from baselines.dsmc.arch import PolicyNetwork


@struct.dataclass
class DSMCConfig:
    # Environment settings
    seed: int = 0
    env_id: str = "cartpole"

    # Algorithm hyperparameters
    num_planner_steps: int = 10
    num_planner_particles: int = 32
    num_belief_particles: int = 32
    total_time_steps: int = 25000
    buffer_size: int = 100000
    learning_starts: int = 5000
    policy_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005


class PlanState(NamedTuple):
    states: Array
    actions: Array
    time_idxs: Array
    log_weights: Array
    weights: Array
    resampling_indices: Array
    done_flags: Array


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    particles: Array,
    weights: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain,
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, particles, weights)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return action, log_prob, bijector.forward(mean)
