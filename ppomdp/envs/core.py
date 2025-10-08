from typing import Callable, NamedTuple

from jax import Array
from distrax import Distribution

from ppomdp.core import Carry, BeliefState
from ppomdp.core import TransitionModel, ObservationModel, RewardFn


class POMDPEnv(NamedTuple):
    num_envs: int
    state_dim: int
    action_dim: int
    obs_dim: int
    num_time_steps: int
    init_dist: Distribution
    belief_prior: Distribution
    trans_model: TransitionModel
    obs_model: ObservationModel
    reward_fn: RewardFn
    feature_fn: Callable


class POMDPState(NamedTuple):
    states: Array
    carry: list[Carry]
    observations: Array
    belief_states: BeliefState
    actions: Array
    next_states: Array
    next_carry: list[Carry]
    next_observations: Array
    next_belief_states: BeliefState
    rewards: Array
    total_rewards: Array
    time_idxs: Array
    done_flags: Array


class QMDPState(NamedTuple):
    states: Array
    carry: list[Carry]
    observations: Array
    actions: Array
    next_states: Array
    next_carry: list[Carry]
    next_observations: Array
    rewards: Array
    time_idxs: Array
    done_flags: Array
