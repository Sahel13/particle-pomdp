from typing import Callable, NamedTuple

from jax import Array
from distrax import Distribution

from ppomdp.core import Carry, InnerState
from ppomdp.core import TransitionModel, ObservationModel, RewardFn


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
    states: Array
    actions: Array
    next_states: Array
    rewards: Array
    total_rewards: Array
    time_steps: Array
    done_flags: Array


class POMDPState(NamedTuple):
    state: Array
    carry: list[Carry]
    observation: Array
    belief: InnerState
    action: Array
    next_state: Array
    next_carry: list[Carry]
    next_observation: Array
    next_belief: InnerState
    reward: Array
    total_reward: Array
    time: Array
    done: Array


class QMDPState(NamedTuple):
    state: Array
    carry: list[Carry]
    observation: Array
    action: Array
    next_state: Array
    next_carry: list[Carry]
    next_observation: Array
    time: Array
    reward: Array
    done: Array
