from typing import Any, Dict, Union, Callable, NamedTuple

import chex

from jax import Array
from distrax import Distribution
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from ppomdp.core import (
    ObservationModel,
    TransitionModel,
    RewardFn
)

Parameters = Union[Dict[str, Any], FrozenDict[str, Any]]
PRNGKey = chex.PRNGKey


class Env(NamedTuple):
    num_envs: int
    state_dim: int
    action_dim: int
    obs_dim: int
    num_time_steps: int
    prior_dist: Distribution
    trans_model: TransitionModel
    obs_model: ObservationModel
    reward_fn: RewardFn
    feature_fn: Callable


class EnvState(NamedTuple):
    state: Array
    action: Array
    next_state: Array
    reward: Array
    total_reward: Array
    time: Array
    done: Array


class JointTrainState(NamedTuple):
    policy_state: TrainState
    critic_state: TrainState
    critic_target_params: Dict


class Config(NamedTuple):
    seed: int = 1
    total_timesteps: int = int(1e5)
    buffer_size: int = int(1e5)
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 3e-4
    critic_lr: float = 1e-3
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005
