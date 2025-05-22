from typing import NamedTuple, Callable
from functools import partial

import jax

from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import Chain, MultivariateNormalDiag, Transformed
from distrax import Distribution

from ppomdp.core import (
    PRNGKey,
    Parameters,
    TransitionModel,
    ObservationModel,
)

from ppomdp.utils import custom_split

from baselines.dsmc.arch import PolicyNetwork


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
    params: Parameters,
    network: PolicyNetwork,
    bijector: Chain,
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, particles)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return action, log_prob, bijector.forward(mean)


@partial(
    jax.jit,
    static_argnames=(
        "num_time_steps",
        "num_trajectory_samples",
        "num_belief_particles",
        "init_dist",
        "belief_prior",
        "trans_model",
        "obs_model",
        "reward_fn",
    )
)
def policy_evaluation(
    rng_key: PRNGKey,
    num_time_steps: int,
    num_trajectory_samples: int,
    num_belief_particles: int,
    init_dist: Distribution,
    belief_prior: Distribution,
    policy_state: TrainState,
    trans_model: TransitionModel,
    obs_model: ObservationModel,
    reward_fn: Callable,
):

    from ppomdp.smc.utils import initialize_belief, update_belief

    def body(carry, key):
        states, beliefs, time_idx = carry

        # Sample actions using the policy
        key, action_key = random.split(key)
        _, _, actions = policy_state.apply_fn(
            rng_key=action_key,
            particles=beliefs.particles,
            params=policy_state.params
        )

        # Compute rewards
        rewards = jax.vmap(reward_fn, (0, 0, None))(states, actions, time_idx)

        # Sample next states
        key, state_keys = custom_split(key, num_trajectory_samples + 1)
        next_states = jax.vmap(trans_model.sample)(state_keys, states, actions)

        # Sample observations
        key, obs_keys = custom_split(key, num_trajectory_samples + 1)
        next_observations = jax.vmap(obs_model.sample)(obs_keys, next_states)

        # Update beliefs
        belief_keys = random.split(key, num_trajectory_samples)
        next_beliefs = jax.vmap(update_belief, (0, None, None, 0, 0, 0))(
            belief_keys, trans_model, obs_model, beliefs, next_observations, actions
        )

        return (next_states, next_beliefs, time_idx + 1), \
            (next_states, actions, next_beliefs, rewards)

    # Initialize
    key, state_key = random.split(rng_key)
    init_states = init_dist.sample(seed=state_key, sample_shape=num_trajectory_samples)
    key, obs_keys = custom_split(key, num_trajectory_samples + 1)
    init_observations = jax.vmap(obs_model.sample)(obs_keys, init_states)

    # Initialize beliefs
    key, belief_keys = custom_split(key, num_trajectory_samples + 1)
    init_beliefs = jax.vmap(initialize_belief, in_axes=(0, None, None, 0, None))(
        belief_keys, belief_prior, obs_model, init_observations, num_belief_particles
    )

    _, (states, actions, beliefs, rewards) = jax.lax.scan(
        f=body,
        init=(init_states, init_beliefs, 0),
        xs=random.split(key, num_time_steps + 1),
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    states = concat_trees(init_states,  states)
    beliefs = concat_trees(init_beliefs, beliefs)
    return rewards, states, actions, beliefs
