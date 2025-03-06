from collections.abc import Callable
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Numeric, PRNGKey
from distrax import Chain, ScalarAffine
from flax.core import FrozenDict
from jax import Array

from baselines.sac.utils import MLP
from ppomdp.bijector import Tanh
from ppomdp.core import InnerState
from ppomdp.policy import squash_policy


class OuterState(NamedTuple):
    """The joint state for environment simulation."""

    states: Array
    belief_states: InnerState
    actions: Array
    next_states: Array
    next_belief_states: InnerState
    rewards: Array
    episodic_rewards: Array
    time_steps: Array
    dones: Array


class Transition(NamedTuple):
    """Named tuple to hold training data."""

    belief_states: tuple[Array, Array]
    actions: Array
    next_belief_states: tuple[Array, Array]
    time_steps: Array
    rewards: Array
    dones: Array


@jax.jit
def get_transition(os: OuterState) -> Transition:
    """Get the transition data for training."""

    particle_set = (os.belief_states.particles, os.belief_states.weights)
    next_particle_set = (os.next_belief_states.particles, os.next_belief_states.weights)
    return Transition(
        belief_states=particle_set,
        actions=os.actions,
        next_belief_states=next_particle_set,
        time_steps=os.time_steps,
        rewards=os.rewards,
        dones=os.dones,
    )


class QNetworks(nn.Module):
    feature_fn: Callable[[Array], Array]
    num_time_steps: int
    recur_size: int
    n_critics: int = 2
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, particles: Array, weights: Array, actions: Array, time: Array):
        # Encode the belief state into a fixed-size vector.
        # TODO: Check if you need to add linear layers before the RNN.
        particles = self.feature_fn(particles)
        particle_set = jnp.concatenate([particles, weights[..., None]], -1)
        encoder_outputs = nn.RNN(nn.GRUCell(self.recur_size))(particle_set)
        encoder_outputs = jnp.take(
            encoder_outputs, -1, axis=-2
        )  # Take the final output.

        norm_time = time / self.num_time_steps
        hidden = jnp.concatenate([encoder_outputs, actions, norm_time[..., None]], -1)

        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_sizes + (1,),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class PolicyNetwork(nn.Module):
    action_dim: int
    feature_fn: Callable
    recur_size: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, particles: Array, weights: Array) -> Array:
        log_std = self.param("log_std", self.init_log_std, self.action_dim)

        # Encode the belief state into a fixed-size vector.
        # TODO: Check if you need to add linear layers before the RNN.
        particles = self.feature_fn(particles)
        particle_set = jnp.concatenate([particles, weights[..., None]], -1)
        encoder_outputs = nn.RNN(nn.GRUCell(self.recur_size))(particle_set)
        encoder_outputs = jnp.take(
            encoder_outputs, -1, axis=-2
        )  # Take the final output.

        return MLP(layer_sizes=self.hidden_sizes + (self.action_dim,))(encoder_outputs)


def sample_and_log_prob(
    rng_key: PRNGKey,
    params: FrozenDict,
    particles: Array,
    weights: Array,
    network: PolicyNetwork,
    action_scale: Numeric,
    action_shift: Numeric,
) -> tuple[Array, Array, Array]:
    """Sample actions and compute their log probabilities."""
    network_out = network.apply({"params": params}, particles, weights)
    bijector = Chain([ScalarAffine(action_shift, action_scale), Tanh()])
    dist = squash_policy(network_out, params["log_std"], bijector)
    sampled_action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    mean_action = bijector.forward(network_out)
    return sampled_action, log_prob, mean_action
