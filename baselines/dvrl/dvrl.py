from copy import deepcopy
from functools import partial

import jax
import optax

from jax import Array, random, numpy as jnp
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.core import BeliefState, PRNGKey
from ppomdp.envs.core import POMDPEnv, POMDPState
from ppomdp.bijector import Tanh
from ppomdp.utils import custom_split
from ppomdp.smc.utils import initialize_belief, update_belief

from baselines.common import JointTrainState, sample_random_actions
from baselines.dvrl.arch import CriticNetwork, PolicyNetwork
from baselines.dvrl.utils import policy_sample_and_log_prob


def _pomdp_base(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    states: Array,
    belief_states: BeliefState,
    random_actions: bool,
) -> tuple[Array, Array, BeliefState, Array]:
    """Sample actions and get the next states and observations."""

    # Sample actions.
    key, action_key = random.split(rng_key)
    if random_actions:
        actions = sample_random_actions(action_key, env_obj)
    else:
        actions, _, _ = policy_state.apply_fn(
            rng_key=action_key,
            params=policy_state.params,
            particles=belief_states.particles,
            weights=belief_states.weights,
        )

    # Get the next states.
    key, state_keys = custom_split(key, env_obj.num_envs + 1)
    next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

    # Sample observations.
    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)

    # Update the belief states.
    key, belief_keys = custom_split(key, env_obj.num_envs + 1)
    next_belief_states = jax.vmap(update_belief, (0, None, None, 0, 0, 0))(
        belief_keys,
        env_obj.trans_model,
        env_obj.obs_model,
        belief_states,
        next_observations,
        actions,
    )

    return next_states, next_observations, next_belief_states, actions


def pomdp_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    num_belief_particles: int,
    random_actions: bool = False,
) -> POMDPState:
    """Initialize the history state."""
    key, prior_key = random.split(rng_key)
    states = env_obj.init_dist.sample(seed=prior_key, sample_shape=(env_obj.num_envs,))

    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

    key, belief_keys = custom_split(key, env_obj.num_envs + 1)
    belief_states = jax.vmap(initialize_belief, (0, None, None, 0, None))(
        belief_keys,
        env_obj.belief_prior,
        env_obj.obs_model,
        observations,
        num_belief_particles,
    )

    key, step_key = random.split(key, 2)
    next_states, next_observations, next_belief_states, actions = _pomdp_base(
        rng_key=step_key,
        env_obj=env_obj,
        policy_state=policy_state,
        states=states,
        belief_states=belief_states,
        random_actions=random_actions,
    )

    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)
    rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
    done_flags = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    return POMDPState(
        states=states,
        carry=[],
        observations=observations,
        belief_states=belief_states,
        actions=actions,
        next_states=next_states,
        next_observations=next_observations,
        next_carry=[],
        next_belief_states=next_belief_states,
        rewards=rewards,
        total_rewards=rewards.copy(),
        time_idxs=time_idxs,
        done_flags=done_flags,
    )


def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    pomdp_state: POMDPState,
    num_belief_particles: int,
    random_actions: bool = False,
) -> POMDPState:

    def _true_fn(_pomdp_state):
        time_idxs = _pomdp_state.time_idxs + 1
        states = _pomdp_state.next_states
        observations = _pomdp_state.next_observations
        belief_states = _pomdp_state.next_belief_states

        next_states, next_observations, next_belief_states, actions = _pomdp_base(
            rng_key=rng_key,
            env_obj=env_obj,
            policy_state=policy_state,
            states=states,
            belief_states=belief_states,
            random_actions=random_actions,
        )

        rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
        total_rewards = _pomdp_state.total_rewards + rewards
        done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(jnp.int32)

        return POMDPState(
            states=states,
            carry=[],
            observations=observations,
            belief_states=belief_states,
            actions=actions,
            next_states=next_states,
            next_carry=[],
            next_observations=next_observations,
            next_belief_states=next_belief_states,
            rewards=rewards,
            total_rewards=total_rewards,
            time_idxs=time_idxs,
            done_flags=done_flags,
        )

    def _false_fn(_pomdp_state):
        return pomdp_init(
            rng_key=rng_key,
            env_obj=env_obj,
            policy_state=policy_state,
            num_belief_particles=num_belief_particles,
            random_actions=random_actions,
        )

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0), _true_fn, _false_fn, pomdp_state
    )


def critic_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    next_actions, next_log_probs, _ = train_state.policy_state.apply_fn(
        rng_key=rng_key,
        particles=pomdp_state.next_belief_states.particles,
        weights=pomdp_state.next_belief_states.weights,
        params=train_state.policy_state.params,
    )
    _next_values = train_state.critic_state.apply_fn(
        train_state.critic_target_params,
        pomdp_state.next_belief_states.particles,
        pomdp_state.next_belief_states.weights,
        next_actions,
        pomdp_state.time_idxs + 1,
    )
    next_values = jnp.min(_next_values, axis=-1) - alpha * next_log_probs
    target_values = (
        pomdp_state.rewards + (1 - pomdp_state.done_flags) * gamma * next_values
    )

    def critic_loss(params):
        _values = train_state.critic_state.apply_fn(
            params,
            pomdp_state.belief_states.particles,
            pomdp_state.belief_states.weights,
            pomdp_state.actions,
            pomdp_state.time_idxs,
        )
        _error = _values - jnp.expand_dims(target_values, -1)
        return 0.5 * jnp.mean(jnp.square(_error))

    grad_fn = jax.value_and_grad(critic_loss)
    loss, grads = grad_fn(train_state.critic_state.params)
    _critic_state = train_state.critic_state.apply_gradients(grads=grads)
    train_state = train_state._replace(critic_state=_critic_state)
    return train_state, loss


def policy_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
) -> tuple[JointTrainState, Array]:

    def actor_loss(params):
        actions, log_probs, _ = train_state.policy_state.apply_fn(
            rng_key=rng_key,
            particles=pomdp_state.next_belief_states.particles,
            weights=pomdp_state.next_belief_states.weights,
            params=params,
        )
        values = train_state.critic_state.apply_fn(
            train_state.critic_state.params,
            pomdp_state.belief_states.particles,
            pomdp_state.belief_states.weights,
            actions,
            pomdp_state.time_idxs,
        )
        min_values = jnp.min(values, axis=-1)
        return jnp.mean(alpha * log_probs - min_values)

    grad_fn = jax.value_and_grad(actor_loss)
    loss, grads = grad_fn(train_state.policy_state.params)
    _policy_state = train_state.policy_state.apply_gradients(grads=grads)
    train_state = train_state._replace(policy_state=_policy_state)
    return train_state, loss


def critic_target_update(train_state: JointTrainState, tau: float) -> JointTrainState:
    updated_params = jax.tree.map(
        lambda param, target: tau * param + (1 - tau) * target,
        train_state.critic_state.params,
        train_state.critic_target_params,
    )
    return train_state._replace(critic_target_params=updated_params)


@jax.jit
def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
    gamma: float,
    tau: float,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)

    # Update critic
    train_state, critic_loss = critic_train_step(
        critic_key, train_state, pomdp_state, alpha, gamma
    )

    # Update policy
    train_state, policy_loss = policy_train_step(
        policy_key, train_state, pomdp_state, alpha
    )

    # Update target critic
    train_state = critic_target_update(train_state, tau)
    return train_state, policy_loss, critic_loss


def create_train_state(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_lr: float,
    critic_lr: float,
    num_belief_particles: int,
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:
    policy_network = PolicyNetwork(
        feature_fn=env_obj.feature_fn,
        encoding_dim=32,
        hidden_sizes=(256, 256),
        output_dim=env_obj.action_dim,
    )
    critic_networks = CriticNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        encoding_dim=32,
        hidden_sizes=(256, 256),
        num_critics=2,
    )

    dummy_particles = jnp.empty((1, num_belief_particles, env_obj.state_dim))
    dummy_weights = jnp.empty((1, num_belief_particles))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_time = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_particles, dummy_weights)["params"]
    critic_params = critic_networks.init(
        critic_key, dummy_particles, dummy_weights, dummy_actions, dummy_time
    )
    critic_target_params = jax.tree.map(lambda x: deepcopy(x), critic_params)

    policy_bijector = Block(Tanh(), ndims=1)
    policy_apply_fn = partial(
        policy_sample_and_log_prob, network=policy_network, bijector=policy_bijector
    )
    policy_train_state = TrainState.create(
        apply_fn=policy_apply_fn, params=policy_params, tx=optax.adam(policy_lr)
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_networks.apply, params=critic_params, tx=optax.adam(critic_lr)
    )
    train_state = JointTrainState(
        policy_train_state, critic_train_state, critic_target_params
    )
    return train_state, policy_network, critic_networks


@partial(
    jax.jit,
    static_argnames=(
        "env_obj",
        "num_belief_particles",
        "random_actions",
    ),
)
def pomdp_rollout(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    num_belief_particles: int,
    random_actions: bool,
):
    """Simulate trajectories with the DVRL policy."""
    def body(pomdp_state, key):
        pomdp_state = pomdp_step(
            rng_key=key,
            env_obj=env_obj,
            policy_state=policy_state,
            pomdp_state=pomdp_state,
            num_belief_particles=num_belief_particles,
            random_actions=random_actions,
        )
        return pomdp_state, pomdp_state

    init_key, scan_keys = custom_split(rng_key, env_obj.num_time_steps + 1)
    init_pomdp_state = pomdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        policy_state=policy_state,
        num_belief_particles=num_belief_particles,
        random_actions=random_actions,
    )
    _, pomdp_states = jax.lax.scan(body, init_pomdp_state, scan_keys)

    pomdp_states = jax.tree.map(
        lambda x, y: jnp.concatenate((x[None], y), axis=0),
        init_pomdp_state,
        pomdp_states,
    )
    # Combine time and num_envs axes
    return jax.tree.map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), pomdp_states)
