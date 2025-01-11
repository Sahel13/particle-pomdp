from collections.abc import Callable
from typing import NamedTuple, Union
from functools import partial

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from distrax import Chain, Distribution, ScalarAffine
from flax.core import FrozenDict
from jax import Array

from ppomdp.bijector import Tanh
from ppomdp.core import InnerState, ObservationModel, RewardFn, TransitionModel
from ppomdp.policy import GRU, LSTM, Carry, squash_policy
from ppomdp.utils import systematic_resampling


class Environment(NamedTuple):
    num_envs: int
    state_dim: int
    action_dim: int
    obs_dim: int
    num_time_steps: int
    action_scale: chex.Numeric
    action_shift: chex.Numeric
    trans_model: TransitionModel
    obs_model: ObservationModel
    reward_fn: RewardFn
    prior_dist: Distribution
    feature_fn: Callable


class OuterState(NamedTuple):
    """The joint state for environment simulation."""

    states: Array
    observations: Array
    carry: list[Carry]
    belief_states: InnerState
    actions: Array
    next_states: Array
    next_observations: Array
    next_carry: list[Carry]
    next_belief_states: InnerState
    rewards: Array
    episodic_rewards: Array
    time_steps: Array
    dones: Array


class Transition(NamedTuple):
    """Named tuple to hold training data."""

    states: Array
    observations: Array
    carry: list[Carry]
    actions: Array
    next_states: Array
    next_observations: Array
    next_carry: list[Carry]
    rewards: Array
    dones: Array


def one_step_genealogy_tracking(
    rng_key: chex.PRNGKey,
    bs: InnerState,
    next_bs: InnerState,
    num_samples: int,
) -> tuple[Array, Array]:
    resampling_idx = systematic_resampling(rng_key, next_bs.weights, num_samples)
    x_tp1 = next_bs.particles[resampling_idx]
    prev_idx = next_bs.resampling_indices[resampling_idx]
    x_t = bs.particles[prev_idx]
    return x_t, x_tp1


@partial(jax.jit, static_argnames="num_samples")
def get_transition(
    rng_key: chex.PRNGKey, os: OuterState, num_samples: int
) -> Transition:
    """Get the transition data for training."""
    num_trajs = os.belief_states.particles.shape[0]
    keys = jax.random.split(rng_key, num_trajs)
    states, next_states = jax.vmap(one_step_genealogy_tracking, (0, 0, 0, None))(
        keys, os.belief_states, os.next_belief_states, num_samples
    )
    return Transition(
        states=states,
        observations=os.observations,
        carry=os.carry,
        actions=os.actions,
        next_states=next_states,
        next_observations=os.next_observations,
        next_carry=os.next_carry,
        rewards=os.rewards,
        dones=os.dones,
    )


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, data: Array):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(hidden_size)(hidden)
            if i != len(self.layer_sizes) - 1:
                hidden = nn.relu(hidden)
        return hidden


class QNetworks(nn.Module):
    n_critics: int = 2
    hidden_sizes: tuple[int, ...] = (256, 256)
    feature_fn: Callable[[Array], Array] = lambda x: x

    @nn.compact
    def __call__(self, states: Array, actions: Array):
        observations = self.feature_fn(states)
        hidden = jnp.concatenate([observations, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_sizes + (1,),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


def sample_and_log_prob(
    rng_key: chex.PRNGKey,
    params: FrozenDict,
    carry: list[Carry],
    observations: Array,
    network: Union[LSTM, GRU],
    action_scale: chex.Numeric,
    action_shift: chex.Numeric,
) -> tuple[list[Carry], Array, Array, Array]:
    """Sample actions and compute their log probabilities."""
    # TODO: Decide whether to use heteroscedastic noise or not.
    carry, mean = network.apply({"params": params}, carry, observations)
    bijector = Chain([ScalarAffine(action_scale, action_shift), Tanh()])
    dist = squash_policy(mean, params["log_std"], bijector)
    sampled_action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    mean_action = bijector.forward(mean)
    return carry, sampled_action, log_prob, mean_action
