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

from baselines.dsmc.arch import CriticNetwork, PolicyNetwork
from baselines.dsmc.utils import (
    PlanState,
    belief_init,
    belief_update,
    policy_sample_and_log_prob,
    sample_random_actions,
    sample_hidden_states
)
from ppomdp.bijector import Tanh
from ppomdp.core import BeliefState, PRNGKey
from ppomdp.envs.core import POMDPEnv, POMDPState
from ppomdp.utils import (
    custom_split,
    propagate_belief,
    resample_belief,
    systematic_resampling,
)


class DSMCConfig(NamedTuple):
    num_belief_particles: int = 32
    num_planner_particles: int = 32
    num_planner_steps: int = 10
    total_timesteps: int = int(25e3)
    buffer_size: int = int(1e5)
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 3e-4
    critic_lr: float = 1e-3
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005


class JointTrainState(NamedTuple):
    policy_state: TrainState
    critic_state: TrainState
    critic_target_params: Dict


def advantage_fn(
    states: Array,
    actions: Array,
    rewards: Array,
    next_states: Array,
    next_actions: Array,
    next_log_probs: Array,
    time_idxs: Array,
    done_flags: Array,
    train_state: JointTrainState,
    alpha: float,
    gamma: float,
):
    _next_values = train_state.critic_state.apply_fn(
        train_state.critic_target_params, next_states, next_actions, time_idxs
    )
    next_values = _next_values - alpha * next_log_probs
    target_values = (1 - done_flags) * gamma * next_values + rewards

    values = train_state.critic_state.apply_fn(
        train_state.critic_state.params, states, actions, time_idxs - 1
    )
    return target_values - values


def planner_trace(rng_key: PRNGKey, plan_states: PlanState) -> Array:
    _, num_particles, _ = plan_states.actions.shape

    resampling_idx = systematic_resampling(rng_key, plan_states.weights[-1], num_particles)
    last_action_particles = plan_states.actions[-1, resampling_idx]

    # Trace the genealogy for the actions.
    def tracing_fn(carry, args):
        idx = carry
        particles, resampling_indices = args
        a = resampling_indices[idx]
        ancestors = particles[a]
        return a, ancestors

    _, action_particles = jax.lax.scan(
        tracing_fn,
        resampling_idx,
        (
            plan_states.actions[:-1],
            plan_states.resampling_indices[1:],
        ),
        reverse=True,
    )

    return jnp.concatenate([action_particles, last_action_particles[None, ...]], axis=0)


def planner_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    init_time_idx: Array,
    belief_state: BeliefState,
    train_state: JointTrainState,
):
    num_planner_particles = alg_cfg.num_planner_particles
    num_belief_particles = alg_cfg.num_belief_particles

    key, sub_key = random.split(rng_key)
    belief_state = resample_belief(sub_key, belief_state, systematic_resampling)

    # duplicate states from belief particles
    states = jnp.repeat(belief_state.particles[None, ...], num_planner_particles, axis=0)

    key, sub_key = random.split(key)
    actions, _, _ = train_state.policy_state.apply_fn(
        rng_key=sub_key,
        params=train_state.policy_state.params,
        particles=states,
        weights=jnp.ones((num_planner_particles, num_belief_particles)) / num_belief_particles,
    )

    time_idxs = init_time_idx * jnp.ones((num_planner_particles,), dtype=jnp.int32)
    done_flags = jnp.zeros((num_planner_particles,), dtype=jnp.int32)

    log_weights = jnp.zeros((num_planner_particles,))
    weights = jnp.ones((num_planner_particles,))
    resampling_indices = jnp.zeros((num_planner_particles,), dtype=jnp.int32)

    return PlanState(
        states=states,
        actions=actions,
        time_idxs=time_idxs,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_indices,
        done_flags=done_flags,
    )


def planner_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    plan_state: PlanState,
    train_state: JointTrainState,
):
    vmap_propagate_belief = jax.vmap(
        propagate_belief,
        in_axes=(0, None, 0, 0)
    )
    vmap_reward_fn = jax.vmap(
        jax.vmap(env_obj.reward_fn, in_axes=(0, None, None)),
        in_axes=(0, 0, 0)
    )
    vmap_advantage_fn = jax.vmap(
        jax.vmap(
            advantage_fn,
            in_axes=(0, None, 0, 0, None, None, None, None, None, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
    )

    num_planner_particles = alg_cfg.num_planner_particles
    num_belief_particles = alg_cfg.num_belief_particles

    # resampling step
    key, sub_key = random.split(rng_key)
    resampling_idx = systematic_resampling(sub_key, plan_state.weights, num_planner_particles)
    _states = plan_state.states[resampling_idx]
    _actions = plan_state.actions[resampling_idx]
    _time_idxs = plan_state.time_idxs[resampling_idx]
    _log_weights = jnp.zeros((num_planner_particles,))
    _weights = jnp.ones((num_planner_particles,))

    # compute rewards
    rewards = vmap_reward_fn(_states, _actions, _time_idxs)

    # sample next states
    action_key, state_keys = custom_split(key, num_planner_particles + 1)
    next_states = vmap_propagate_belief(state_keys, env_obj.trans_model, _states, _actions)

    # set done flags
    time_idxs = _time_idxs + 1
    done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(jnp.int32)

    # sample next actions
    next_actions, next_log_probs, _ = train_state.policy_state.apply_fn(
        rng_key=action_key,
        params=train_state.policy_state.params,
        particles=next_states,
        weights=jnp.ones((num_planner_particles, num_belief_particles)) / num_belief_particles,
    )

    # reweight with advantage
    log_potentials = vmap_advantage_fn(
        _states,
        _actions,
        rewards,
        next_states,
        next_actions,
        next_log_probs,
        time_idxs,
        done_flags,
        train_state,
        alg_cfg.alpha,
        alg_cfg.gamma,
    )
    log_weights = jnp.mean(log_potentials, axis=1)[..., -1] + _log_weights
    weights = jax.nn.softmax(log_weights)

    return PlanState(
        states=next_states,
        actions=next_actions,
        time_idxs=time_idxs,
        log_weights=log_weights,
        weights=weights,
        resampling_indices=resampling_idx,
        done_flags=done_flags,
    )


def planner_step_dummy(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    plan_state: PlanState,
    train_state: JointTrainState,
):
    num_planner_particles = alg_cfg.num_planner_particles

    return PlanState(
        states=plan_state.states,
        actions=plan_state.actions,
        time_idxs=plan_state.time_idxs,
        log_weights=plan_state.log_weights,
        weights=plan_state.weights,
        resampling_indices=jnp.arange(num_planner_particles, dtype=jnp.int32),
        done_flags=plan_state.done_flags,
    )


def planner_run(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    init_time_idx: Array,
    belief_state: BeliefState,
    train_state: JointTrainState,
):
    def planner_loop(carry, _):
        plan_state, key = carry
        key, step_key = random.split(key)

        def _true_fn(_plan_state):
            return planner_step(
                rng_key=step_key,
                env_obj=env_obj,
                alg_cfg=alg_cfg,
                plan_state=_plan_state,
                train_state=train_state,
            )

        def _false_fn(_plan_state):
            return planner_step_dummy(
                rng_key=step_key,
                env_obj=env_obj,
                alg_cfg=alg_cfg,
                plan_state=_plan_state,
                train_state=train_state,
            )

        next_plan_state = jax.lax.cond(
            jnp.all(plan_state.done_flags == 0), _true_fn, _false_fn, plan_state
        )
        return (next_plan_state, key), next_plan_state

    key, init_key, scan_key = random.split(rng_key, 3)
    init_plan_state = planner_init(
        rng_key=init_key,
        env_obj=env_obj,
        alg_cfg=alg_cfg,
        init_time_idx=init_time_idx,
        belief_state=belief_state,
        train_state=train_state,
    )

    _, plan_states = jax.lax.scan(
        planner_loop, (init_plan_state, scan_key), length=alg_cfg.num_planner_steps
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    plan_states = concat_trees(init_plan_state, plan_states)

    trace_key, select_key = random.split(key)
    traced_actions = planner_trace(trace_key, plan_states)
    idx = random.choice(select_key, alg_cfg.num_planner_particles)
    return traced_actions[0, idx, ...]  # select random action from zero-th time step


def _pomdp_base(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    states: Array,
    belief_states: BeliefState,
    time_idxs: Array,
    train_state: JointTrainState,
    random_actions: bool,
) -> tuple[Array, Array, BeliefState, Array]:
    # Sample action.
    if random_actions:
        key, action_key = random.split(rng_key, 2)
        actions = sample_random_actions(action_key, env_obj)
    else:
        key, action_keys = custom_split(rng_key, env_obj.num_envs + 1)
        actions = jax.vmap(planner_run, in_axes=(0, None, None, 0, 0, None))(
            action_keys,
            env_obj,
            alg_cfg,
            time_idxs,
            belief_states,
            train_state,
        )

    # Sample next state.
    key, state_keys = custom_split(key, env_obj.num_envs + 1)
    next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

    # Sample observation.
    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)

    # Update belief.
    key, belief_keys = custom_split(key, env_obj.num_envs + 1)
    next_belief_states = jax.vmap(belief_update, (0, None, 0, 0, 0))(
        belief_keys, env_obj, belief_states, next_observations, actions
    )
    return next_states, next_observations, next_belief_states, actions


def pomdp_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    train_state: JointTrainState,
    random_actions: bool = False,
) -> POMDPState:
    key, prior_key = random.split(rng_key, 2)
    states = env_obj.prior_dist.sample(seed=prior_key, sample_shape=(env_obj.num_envs,))

    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

    key, belief_keys = custom_split(key, env_obj.num_envs + 1)
    belief_states = jax.vmap(belief_init, (0, None, 0, None))(
        belief_keys, env_obj, observations, alg_cfg.num_belief_particles
    )

    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    key, step_key = random.split(key, 2)
    next_states, next_observations, next_belief_states, actions = _pomdp_base(
        rng_key=step_key,
        env_obj=env_obj,
        alg_cfg=alg_cfg,
        states=states,
        belief_states=belief_states,
        time_idxs=time_idxs,
        train_state=train_state,
        random_actions=random_actions,
    )

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


@partial(jax.jit, static_argnums=(1, 2, 5), donate_argnames="pomdp_state")
def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    pomdp_state: POMDPState,
    train_state: JointTrainState,
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
            alg_cfg=alg_cfg,
            states=states,
            belief_states=belief_states,
            time_idxs=time_idxs,
            train_state=train_state,
            random_actions=random_actions,
        )

        rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
        total_rewards = _pomdp_state.total_rewards + rewards
        done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(
            jnp.int32
        )

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
        return pomdp_init(rng_key, env_obj, alg_cfg, train_state, random_actions)

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0), _true_fn, _false_fn, pomdp_state
    )


def create_train_state(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:
    policy_log_std = jnp.ones(env_obj.action_dim)
    policy_network = PolicyNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        recurr_size=32,
        hidden_sizes=(256, 256),
        output_dim=env_obj.action_dim,
        init_log_std=constant(policy_log_std),
    )
    critic_network = CriticNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        hidden_sizes=(256, 256),
    )

    dummy_particles = jnp.empty((1, alg_cfg.num_planner_particles, env_obj.state_dim))
    dummy_weights = jnp.empty((1, alg_cfg.num_planner_particles))
    dummy_states = jnp.empty((1, env_obj.state_dim))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_time = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_particles, dummy_weights)["params"]
    critic_params = critic_network.init(critic_key, dummy_states, dummy_actions, dummy_time)
    critic_target_params = jax.tree.map(lambda x: deepcopy(x), critic_params)

    policy_bijector = Block(Tanh(), ndims=1)
    policy_apply_fn = partial(
        policy_sample_and_log_prob,
        network=policy_network,
        bijector=policy_bijector
    )
    policy_train_state = TrainState.create(
        apply_fn=policy_apply_fn,
        params=policy_params,
        tx=optax.adam(alg_cfg.policy_lr)
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_params,
        tx=optax.adam(alg_cfg.critic_lr),
    )
    train_state = JointTrainState(
        policy_train_state,
        critic_train_state,
        critic_target_params
    )
    return train_state, policy_network, critic_network


def critic_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    key, sub_key = random.split(rng_key)
    next_actions, next_log_probs, _ = train_state.policy_state.apply_fn(
        rng_key=sub_key,
        params=train_state.policy_state.params,
        particles=pomdp_state.next_belief_states.particles,
        weights=pomdp_state.next_belief_states.weights,
    )

    key, state_key, next_state_key = random.split(key, 3)
    _next_states = sample_hidden_states(
        state_key,
        pomdp_state.next_belief_states.particles,
        pomdp_state.next_belief_states.weights
    )
    _states = sample_hidden_states(
        next_state_key,
        pomdp_state.belief_states.particles,
        pomdp_state.belief_states.weights
    )

    _next_values = train_state.critic_state.apply_fn(
        train_state.critic_target_params,
        _next_states,
        next_actions,
        pomdp_state.time_idxs + 1,
    )
    next_values = jnp.min(_next_values, axis=-1) - alpha * next_log_probs
    target_values = pomdp_state.rewards + (1 - pomdp_state.done_flags) * gamma * next_values

    def critic_loss(params):
        _values = train_state.critic_state.apply_fn(
            params,
            _states,
            pomdp_state.actions,
            pomdp_state.time_idxs
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
        key, sub_key = random.split(rng_key)
        actions, log_probs, _ = train_state.policy_state.apply_fn(
            rng_key=sub_key,
            params=params,
            particles=pomdp_state.belief_states.particles,
            weights=pomdp_state.belief_states.weights,
        )

        key, sub_key = random.split(key)
        _states = sample_hidden_states(
            key,
            pomdp_state.belief_states.particles,
            pomdp_state.belief_states.weights,
        )

        values = train_state.critic_state.apply_fn(
            train_state.critic_state.params,
            _states,
            actions,
            pomdp_state.time_idxs
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


def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alg_cfg: DSMCConfig,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)
    train_state, critic_loss = critic_train_step(critic_key, train_state, pomdp_state, alg_cfg.alpha, alg_cfg.gamma)
    train_state, policy_loss = policy_train_step(policy_key, train_state, pomdp_state, alg_cfg.alpha)
    train_state = critic_target_update(train_state, alg_cfg.tau)
    return train_state, policy_loss, critic_loss


@partial(jax.jit, static_argnums=(1, 2, 4, 7))
def step_and_train(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    alg_cfg: DSMCConfig,
    pomdp_state: POMDPState,
    buffer_obj: UniformSamplingQueue,
    buffer_state: ReplayBufferState,
    train_state: JointTrainState,
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
            train_state=_train_state,
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
