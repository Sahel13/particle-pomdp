from typing import Any, Dict, Union, Callable, NamedTuple

import chex
from flax.core import FrozenDict
from distrax import Distribution
from jax import Array

from ppomdp.core import TransitionModel, ObservationModel, RewardFn

PRNGKey = chex.PRNGKey
Parameters = Union[Dict[str, Any], FrozenDict[str, Any]]


class MDPEnv(NamedTuple):
    num_envs: int
    state_dim: int
    action_dim: int
    num_time_steps: int
    prior_dist: Distribution
    trans_model: TransitionModel
    reward_fn: RewardFn
    feature_fn: Callable


class POMDPEnv(NamedTuple):
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


class MDPState(NamedTuple):
    state: Array
    action: Array
    next_state: Array
    reward: Array
    total_reward: Array
    time: Array
    done: Array
