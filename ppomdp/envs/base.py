from collections.abc import Callable
from typing import NamedTuple

from chex import Numeric
from distrax import Distribution

from ppomdp.core import ObservationModel, RewardFn, TransitionModel


class Environment(NamedTuple):
    num_envs: int
    state_dim: int
    action_dim: int
    obs_dim: int
    num_time_steps: int
    action_scale: float
    action_shift: float
    trans_model: TransitionModel
    obs_model: ObservationModel
    reward_fn: RewardFn
    prior_dist: Distribution
    feature_fn: Callable
