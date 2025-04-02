from copy import deepcopy
from functools import partial
from typing import Dict, NamedTuple

import jax
import optax
from brax.training.replay_buffers import ReplayBufferState, UniformSamplingQueue
from distrax import Block
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import Array, random
from jax import numpy as jnp

from baselines.slac.arch import CriticNetwork, PolicyNetwork
from baselines.slac.utils import (
    belief_init,
    belief_update,
    policy_sample_and_log_prob,
    sample_random_actions,
)
from ppomdp.arch import GRUEncoder, MLPDecoder
from ppomdp.bijector import Tanh
from ppomdp.core import BeliefState, Carry, PRNGKey
from ppomdp.envs.core import POMDPEnv, POMDPState
from ppomdp.utils import custom_split


class SLACConfig(NamedTuple):
    num_belief_particles: int = 32
    total_timesteps: int = int(1e5)
    buffer_size: int = int(1e6)
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 1e-4
    critic_lr: float = 1e-3
    alpha: float = 0.2
    gamma: float = 0.995
    tau: float = 0.005


class JointTrainState(NamedTuple):
    policy_state: TrainState
    critic_state: TrainState
    critic_target_params: Dict


def _pomdp_base(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: SLACConfig,
    states: Array,
    carry: list[Carry],
    observations: Array,
    belief_states: BeliefState,
    policy_state: TrainState,
    random_actions: bool,
) -> tuple[Array, list[Carry], Array, BeliefState, Array]:
    """Sample actions and get the next states and observations."""

    # Sample action.
    key, action_key = random.split(rng_key, 2)
    if random_actions:
        next_carry = carry
        actions = sample_random_actions(action_key, env_obj)
    else:
        next_carry, actions, _, _ = policy_state.apply_fn(
            rng_key=action_key,
            params=policy_state.params,
            carry=carry,
            observation=observations,
        )

    # Sample next state.
    key, state_keys = custom_split(key, env_obj.num_envs + 1)
    next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

    # Sample observation.
    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)

    # Update belief.
    belief_keys = random.split(key, env_obj.num_envs)
    next_belief_states = jax.vmap(belief_update, (0, None, 0, 0, 0))(
        belief_keys, env_obj, belief_states, next_observations, actions
    )

    return next_states, next_carry, next_observations, next_belief_states, actions


def pomdp_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: SLACConfig,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    random_actions: bool = False,
) -> POMDPState:
    """Initialize the env state."""
    key, prior_key = random.split(rng_key, 2)
    states = env_obj.prior_dist.sample(seed=prior_key, sample_shape=(env_obj.num_envs,))

    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

    key, belief_keys = custom_split(key, env_obj.num_envs + 1)
    belief_states = jax.vmap(belief_init, (0, None, 0, None))(
        belief_keys, env_obj, observations, alg_cfg.num_belief_particles
    )

    carry = policy_network.reset(env_obj.num_envs)

    key, step_key = random.split(key, 2)
    next_states, next_carry, next_observations, next_belief_states, actions = (
        _pomdp_base(
            rng_key=step_key,
            env_obj=env_obj,
            alg_cfg=alg_cfg,
            states=states,
            carry=carry,
            observations=observations,
            belief_states=belief_states,
            policy_state=policy_state,
            random_actions=random_actions,
        )
    )

    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)
    rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
    done_flags = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    return POMDPState(
        states=states,
        carry=carry,
        observations=observations,
        belief_states=belief_states,
        actions=actions,
        next_states=next_states,
        next_observations=next_observations,
        next_carry=next_carry,
        next_belief_states=next_belief_states,
        rewards=rewards,
        total_rewards=rewards.copy(),
        time_idxs=time_idxs,
        done_flags=done_flags,
    )


@partial(
    jax.jit,
    static_argnames=("env_obj", "alg_cfg", "policy_network", "random_actions"),
    donate_argnames="pomdp_state",
)
def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: SLACConfig,
    pomdp_state: POMDPState,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    random_actions: bool = False,
) -> POMDPState:
    def _true_fn(_pomdp_state):
        time_idxs = _pomdp_state.time_idxs + 1
        states = _pomdp_state.next_states
        carry = _pomdp_state.next_carry
        observations = _pomdp_state.next_observations
        belief_states = _pomdp_state.next_belief_states

        next_states, next_carry, next_observations, next_belief_states, actions = (
            _pomdp_base(
                rng_key=rng_key,
                env_obj=env_obj,
                alg_cfg=alg_cfg,
                states=states,
                carry=carry,
                observations=observations,
                belief_states=belief_states,
                policy_state=policy_state,
                random_actions=random_actions,
            )
        )

        rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
        total_rewards = _pomdp_state.total_rewards + rewards
        done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(
            jnp.int32
        )

        return POMDPState(
            states=states,
            carry=carry,
            observations=observations,
            belief_states=belief_states,
            actions=actions,
            next_states=next_states,
            next_carry=next_carry,
            next_observations=next_observations,
            next_belief_states=next_belief_states,
            rewards=rewards,
            total_rewards=total_rewards,
            time_idxs=time_idxs,
            done_flags=done_flags,
        )

    def _false_fn(_pomdp_state):
        return pomdp_init(
            rng_key, env_obj, alg_cfg, policy_state, policy_network, random_actions
        )

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0), _true_fn, _false_fn, pomdp_state
    )


def create_train_state(
    rng_key: PRNGKey, env_obj: POMDPEnv, alg_cfg: SLACConfig
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:
    policy_log_std = jnp.ones(env_obj.action_dim)
    policy_encoder = GRUEncoder(
        feature_fn=lambda x: x,
        encoder_size=(256, 256),
        recurr_size=(128, 128),
    )
    policy_decoder = MLPDecoder(
        decoder_size=(256, 256),
        output_dim=env_obj.action_dim,
    )
    policy_network = PolicyNetwork(
        encoder=policy_encoder,
        decoder=policy_decoder,
        init_log_std=constant(policy_log_std),
    )
    critic_networks = CriticNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        hidden_sizes=(256, 256),
        num_critics=2,
    )

    dummy_states = jnp.empty((1, env_obj.state_dim))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_carry = policy_network.reset(env_obj.num_envs)
    dummy_observations = jnp.empty((1, env_obj.obs_dim))
    dummy_time = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_carry, dummy_observations)[
        "params"
    ]
    critic_params = critic_networks.init(
        critic_key, dummy_states, dummy_actions, dummy_time
    )
    critic_target_params = jax.tree.map(lambda x: deepcopy(x), critic_params)

    policy_bijector = Block(Tanh(), ndims=1)
    policy_apply_fn = partial(
        policy_sample_and_log_prob, network=policy_network, bijector=policy_bijector
    )
    policy_train_state = TrainState.create(
        apply_fn=policy_apply_fn, params=policy_params, tx=optax.adam(alg_cfg.policy_lr)
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_networks.apply,
        params=critic_params,
        tx=optax.adam(alg_cfg.critic_lr),
    )
    train_state = JointTrainState(
        policy_train_state, critic_train_state, critic_target_params
    )
    return train_state, policy_network, critic_networks


def sample_hidden_states(rng_key: PRNGKey, particles: Array, weights: Array) -> Array:
    """Sample one hidden state for each belief state.

    `particles` has shape (batch_size, num_particles, state_dim).
    """
    batch_size, num_particles = particles.shape[:2]

    def choice_fn(key, _particles, _weights):
        idx = random.choice(key, a=num_particles, p=_weights)
        return _particles[idx]

    keys = random.split(rng_key, batch_size)
    return jax.vmap(choice_fn)(keys, particles, weights)


def critic_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    key, sub_key = random.split(rng_key)

    _, next_actions, next_log_probs, _ = train_state.policy_state.apply_fn(
        rng_key=sub_key,
        params=train_state.policy_state.params,
        carry=pomdp_state.next_carry,
        observation=pomdp_state.next_observations,
    )

    key, state_key, next_state_key = random.split(key, 3)
    _states = sample_hidden_states(
        state_key,
        pomdp_state.belief_states.particles,
        pomdp_state.belief_states.weights,
    )
    _next_states = sample_hidden_states(
        next_state_key,
        pomdp_state.next_belief_states.particles,
        pomdp_state.next_belief_states.weights,
    )

    _next_values = train_state.critic_state.apply_fn(
        train_state.critic_target_params,
        _next_states,
        next_actions,
        pomdp_state.time_idxs + 1,
    )
    next_values = jnp.min(_next_values, axis=-1) - alpha * next_log_probs
    target_values = (
        pomdp_state.rewards + (1 - pomdp_state.done_flags) * gamma * next_values
    )

    def critic_loss(params):
        _values = train_state.critic_state.apply_fn(
            params, _states, pomdp_state.actions, pomdp_state.time_idxs
        )
        _error = _values - jnp.expand_dims(target_values, -1)
        return 0.5 * jnp.mean(jnp.square(_error))

    grad_fn = jax.value_and_grad(critic_loss)
    loss, grads = grad_fn(train_state.critic_state.params)
    critic_state = train_state.critic_state.apply_gradients(grads=grads)
    train_state = train_state._replace(critic_state=critic_state)
    return train_state, loss


def policy_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
) -> tuple[JointTrainState, Array]:
    def actor_loss(params):
        key, action_key, state_key = random.split(rng_key, 3)
        _, actions, log_probs, _ = train_state.policy_state.apply_fn(
            rng_key=action_key,
            params=params,
            carry=pomdp_state.carry,
            observation=pomdp_state.observations,
        )
        _states = sample_hidden_states(
            state_key,
            pomdp_state.belief_states.particles,
            pomdp_state.belief_states.weights,
        )
        values = train_state.critic_state.apply_fn(
            train_state.critic_state.params, _states, actions, pomdp_state.time_idxs
        )
        min_values = jnp.min(values, axis=-1)
        return jnp.mean(alpha * log_probs - min_values)

    grad_fn = jax.value_and_grad(actor_loss)
    loss, grads = grad_fn(train_state.policy_state.params)
    policy_state = train_state.policy_state.apply_gradients(grads=grads)
    train_state = train_state._replace(policy_state=policy_state)
    return train_state, loss


def critic_target_update(train_state: JointTrainState, tau: float) -> JointTrainState:
    updated_params = jax.tree.map(
        lambda param, target: tau * param + (1 - tau) * target,
        train_state.critic_state.params,
        train_state.critic_target_params,
    )
    return train_state._replace(critic_target_params=updated_params)


def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alg_cfg: SLACConfig,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)
    train_state, critic_loss = critic_train_step(
        critic_key, train_state, pomdp_state, alg_cfg.alpha, alg_cfg.gamma
    )
    train_state, policy_loss = policy_train_step(
        policy_key, train_state, pomdp_state, alg_cfg.alpha
    )
    train_state = critic_target_update(train_state, alg_cfg.tau)
    return train_state, policy_loss, critic_loss


@partial(
    jax.jit,
    static_argnames=("env_obj", "alg_cfg", "buffer_obj", "policy_network", "num_steps"),
    donate_argnames=("buffer_state", "pomdp_state", "train_state"),
)
def step_and_train(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: SLACConfig,
    pomdp_state: POMDPState,
    buffer_obj: UniformSamplingQueue,
    buffer_state: ReplayBufferState,
    train_state: JointTrainState,
    policy_network: PolicyNetwork,
    num_steps: int,
):
    def body(carry, key):
        _pomdp_state, _buffer_state, _train_state = carry
        _step_key, _train_key = random.split(key)

        _pomdp_state = pomdp_step(
            rng_key=_step_key,
            env_obj=env_obj,
            alg_cfg=alg_cfg,
            pomdp_state=_pomdp_state,
            policy_state=_train_state.policy_state,
            policy_network=policy_network,
        )
        _buffer_state = buffer_obj.insert(_buffer_state, _pomdp_state)
        _buffer_state, _pomdp_state_sample = buffer_obj.sample(_buffer_state)

        _train_state, _, _ = gradient_step(
            rng_key=_train_key,
            train_state=_train_state,
            pomdp_state=_pomdp_state_sample,
            alg_cfg=alg_cfg,
        )
        return (_pomdp_state, _buffer_state, _train_state), None

    keys = random.split(rng_key, num_steps)
    (pomdp_state, buffer_state, train_state), _ = jax.lax.scan(
        body, (pomdp_state, buffer_state, train_state), keys
    )
    return pomdp_state, buffer_state, train_state


@partial(jax.jit, static_argnames=("env", "policy", "num_samples"))
def evaluate(
    key: PRNGKey,
    env: POMDPEnv,
    train_state: JointTrainState,
    policy: PolicyNetwork,
    num_samples: int = 100,
):
    """Simulate trajectories using the policy and compute the expected reward."""

    def body(carry, _key):
        states, policy_carry, observations, t = carry

        # Sample actions.
        _key, action_key = random.split(_key)
        policy_carry, _, _, actions = train_state.policy_state.apply_fn(
            action_key, policy_carry, observations, train_state.policy_state.params
        )
        # Compute rewards.
        rewards = jax.vmap(env.reward_fn, (0, 0, None))(states, actions, t)
        # Sample next states.
        _key, state_keys = custom_split(_key, num_samples + 1)
        states = jax.vmap(env.trans_model.sample)(state_keys, states, actions)
        # Sample observations.
        obs_keys = random.split(_key, num_samples)
        observations = jax.vmap(env.obs_model.sample)(obs_keys, states)

        return (states, policy_carry, observations, t + 1), (states, actions, rewards)

    # Initialize.
    key, state_key = random.split(key)
    init_states = env.prior_dist.sample(seed=state_key, sample_shape=num_samples)
    key, obs_keys = custom_split(key, num_samples + 1)
    init_observations = jax.vmap(env.obs_model.sample)(obs_keys, init_states)
    init_policy_carry = policy.reset(num_samples)

    _, (states, actions, rewards) = jax.lax.scan(
        body,
        (init_states, init_policy_carry, init_observations, 0),
        random.split(key, env.num_time_steps + 1),
    )
    states = jnp.concatenate([init_states[None], states], axis=0)
    expected_reward = jnp.mean(jnp.sum(rewards, axis=0))
    return expected_reward, states, actions
