from copy import deepcopy
from functools import partial

import jax
import optax

from jax import Array, numpy as jnp, random
from flax.training.train_state import TrainState
from distrax import Block

from ppomdp.core import BeliefState, PRNGKey
from ppomdp.envs.core import POMDPEnv, POMDPState
from ppomdp.bijector import Tanh
from ppomdp.utils import custom_split
from ppomdp.smc.utils import (
    initialize_belief,
    update_belief,
    propagate_belief,
    resample_belief,
    systematic_resampling,
)

from baselines.common import (
    JointTrainState,
    sample_hidden_states,
    sample_random_actions,
)
from baselines.dsmc.arch import CriticNetwork, PolicyNetwork
from baselines.dsmc.utils import PlanState, policy_sample_and_log_prob


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
        f=tracing_fn,
        init=resampling_idx,
        xs=(
            plan_states.actions[:-1],
            plan_states.resampling_indices[1:],
        ),
        reverse=True,
    )
    return jnp.concatenate([action_particles, last_action_particles[None, ...]], axis=0)


def planner_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    train_state: JointTrainState,
    init_time_idx: Array,
    belief_state: BeliefState,
    num_belief_particles: int,
    num_planner_particles: int,
):
    key, sub_key = random.split(rng_key)
    belief_state = resample_belief(sub_key, belief_state, systematic_resampling)

    # duplicate states from belief particles
    states = jnp.repeat(
        belief_state.particles[None, ...], num_planner_particles, axis=0
    )

    key, sub_key = random.split(key)
    actions, _, _ = train_state.policy_state.apply_fn(
        rng_key=sub_key, particles=states, params=train_state.policy_state.params
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
    train_state: JointTrainState,
    plan_state: PlanState,
    num_planner_particles: int,
    num_belief_particles: int,
    alpha: float,
    gamma: float,
):
    vmap_propagate_belief = jax.vmap(propagate_belief, in_axes=(0, None, 0, 0))
    vmap_reward_fn = jax.vmap(
        jax.vmap(env_obj.reward_fn, in_axes=(0, None, None)), in_axes=(0, 0, 0)
    )
    vmap_advantage_fn = jax.vmap(
        jax.vmap(
            advantage_fn,
            in_axes=(0, None, 0, 0, None, None, None, None, None, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
    )

    # resampling step
    key, sub_key = random.split(rng_key)
    resampling_idx = systematic_resampling(
        sub_key, plan_state.weights, num_planner_particles
    )
    _states = plan_state.states[resampling_idx]
    _actions = plan_state.actions[resampling_idx]
    _time_idxs = plan_state.time_idxs[resampling_idx]
    _log_weights = jnp.zeros((num_planner_particles,))
    _weights = jnp.ones((num_planner_particles,))

    # compute rewards
    rewards = vmap_reward_fn(_states, _actions, _time_idxs)

    # sample next states
    action_key, state_keys = custom_split(key, num_planner_particles + 1)
    next_states = vmap_propagate_belief(
        state_keys, env_obj.trans_model, _states, _actions
    )

    # set done flags
    time_idxs = _time_idxs + 1
    done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(jnp.int32)

    # sample next actions
    next_actions, next_log_probs, _ = train_state.policy_state.apply_fn(
        rng_key=action_key,
        particles=next_states,
        params=train_state.policy_state.params,
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
        alpha,
        gamma,
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
    train_state: JointTrainState,
    plan_state: PlanState,
    num_belief_particles: int,
    num_planner_particles: int,
    alpha: float,
    gamma: float,
):
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
    train_state: JointTrainState,
    belief_state: BeliefState,
    init_time_idx: Array,
    num_planner_steps: int,
    num_planner_particles: int,
    num_belief_particles: int,
    alpha: float,
    gamma: float,
):
    def planner_loop(carry, _):
        plan_state, key = carry
        key, step_key = random.split(key)

        def _true_fn(_plan_state):
            return planner_step(
                rng_key=step_key,
                env_obj=env_obj,
                train_state=train_state,
                plan_state=_plan_state,
                num_planner_particles=num_planner_particles,
                num_belief_particles=num_belief_particles,
                alpha=alpha,
                gamma=gamma,
            )

        def _false_fn(_plan_state):
            return planner_step_dummy(
                rng_key=step_key,
                env_obj=env_obj,
                train_state=train_state,
                plan_state=_plan_state,
                num_planner_particles=num_planner_particles,
                num_belief_particles=num_belief_particles,
                alpha=alpha,
                gamma=gamma,
            )

        next_plan_state = jax.lax.cond(
            jnp.all(plan_state.done_flags == 0), _true_fn, _false_fn, plan_state
        )
        return (next_plan_state, key), next_plan_state

    key, init_key, scan_key = random.split(rng_key, 3)
    init_plan_state = planner_init(
        rng_key=init_key,
        env_obj=env_obj,
        train_state=train_state,
        init_time_idx=init_time_idx,
        belief_state=belief_state,
        num_belief_particles=num_belief_particles,
        num_planner_particles=num_planner_particles,
    )

    _, plan_states = jax.lax.scan(
        planner_loop, (init_plan_state, scan_key), length=num_planner_steps
    )

    def concat_trees(x, y):
        return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

    plan_states = concat_trees(init_plan_state, plan_states)

    trace_key, select_key = random.split(key)
    traced_actions = planner_trace(trace_key, plan_states)
    idx = random.choice(select_key, num_planner_particles)
    return traced_actions[0, idx, ...]  # select random action from zero-th time step


def _pomdp_base(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    train_state: JointTrainState,
    states: Array,
    belief_states: BeliefState,
    time_idxs: Array,
    num_belief_particles: int,
    num_planner_particles: int,
    num_planner_steps: int,
    alpha: float,
    gamma: float,
    random_actions: bool,
) -> tuple[Array, Array, BeliefState, Array]:
    # Sample action.
    if random_actions:
        key, action_key = random.split(rng_key, 2)
        actions = sample_random_actions(action_key, env_obj)
    else:
        key, action_keys = custom_split(rng_key, env_obj.num_envs + 1)
        actions = jax.vmap(
            planner_run, in_axes=(0, None, None, 0, 0, None, None, None, None, None)
        )(
            action_keys,
            env_obj,
            train_state,
            belief_states,
            time_idxs,
            num_planner_steps,
            num_planner_particles,
            num_belief_particles,
            alpha,
            gamma,
        )

    # Sample next state.
    key, state_keys = custom_split(key, env_obj.num_envs + 1)
    next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

    # Sample observation.
    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)

    # Update belief.
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
    train_state: JointTrainState,
    num_belief_particles: int,
    num_planner_particles: int,
    num_planner_steps: int,
    alpha: float,
    gamma: float,
    random_actions: bool = False,
) -> POMDPState:
    key, prior_key = random.split(rng_key, 2)
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

    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    key, step_key = random.split(key, 2)
    next_states, next_observations, next_belief_states, actions = _pomdp_base(
        rng_key=step_key,
        env_obj=env_obj,
        train_state=train_state,
        states=states,
        belief_states=belief_states,
        time_idxs=time_idxs,
        num_belief_particles=num_belief_particles,
        num_planner_particles=num_planner_particles,
        num_planner_steps=num_planner_steps,
        alpha=alpha,
        gamma=gamma,
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


def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    num_belief_particles: int,
    num_planner_particles: int,
    num_planner_steps: int,
    alpha: float,
    gamma: float,
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
            train_state=train_state,
            states=states,
            belief_states=belief_states,
            time_idxs=time_idxs,
            num_belief_particles=num_belief_particles,
            num_planner_particles=num_planner_particles,
            num_planner_steps=num_planner_steps,
            alpha=alpha,
            gamma=gamma,
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
        return pomdp_init(
            rng_key=rng_key,
            env_obj=env_obj,
            train_state=train_state,
            num_belief_particles=num_belief_particles,
            num_planner_particles=num_planner_particles,
            num_planner_steps=num_planner_steps,
            alpha=alpha,
            gamma=gamma,
            random_actions=random_actions,
        )

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0),
        _true_fn,
        _false_fn,
        pomdp_state,
    )


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
        particles=pomdp_state.next_belief_states.particles,
        params=train_state.policy_state.params,
    )

    key, state_key, next_state_key = random.split(key, 3)
    _next_states = sample_hidden_states(
        state_key,
        pomdp_state.next_belief_states.particles,
        pomdp_state.next_belief_states.weights,
    )
    _states = sample_hidden_states(
        next_state_key,
        pomdp_state.belief_states.particles,
        pomdp_state.belief_states.weights,
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
            particles=pomdp_state.belief_states.particles,
            params=params,
        )

        key, sub_key = random.split(key)
        _states = sample_hidden_states(
            key,
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


@partial(jax.jit, donate_argnames="train_state")
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
        rng_key=critic_key,
        train_state=train_state,
        pomdp_state=pomdp_state,
        alpha=alpha,
        gamma=gamma,
    )

    # Update policy
    train_state, policy_loss = policy_train_step(
        rng_key=policy_key,
        train_state=train_state,
        pomdp_state=pomdp_state,
        alpha=alpha,
    )

    # Update target critic
    train_state = critic_target_update(train_state, tau)
    return train_state, critic_loss, policy_loss


def create_train_state(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_lr: float,
    critic_lr: float,
    num_planner_particles: int,
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:

    policy_network = PolicyNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        encoding_dim=32,
        hidden_sizes=(256, 256),
        output_dim=env_obj.action_dim,
    )
    critic_network = CriticNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        hidden_sizes=(256, 256),
    )

    dummy_particles = jnp.empty((1, num_planner_particles, env_obj.state_dim))
    dummy_states = jnp.empty((1, env_obj.state_dim))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_time = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_particles)["params"]
    critic_params = critic_network.init(critic_key, dummy_states, dummy_actions, dummy_time)
    critic_target_params = jax.tree.map(lambda x: deepcopy(x), critic_params)

    policy_bijector = Block(Tanh(), ndims=1)
    policy_apply_fn = partial(
        policy_sample_and_log_prob, network=policy_network, bijector=policy_bijector
    )
    policy_train_state = TrainState.create(
        apply_fn=policy_apply_fn, params=policy_params, tx=optax.adam(policy_lr)
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_params,
        tx=optax.adam(critic_lr),
    )
    train_state = JointTrainState(
        policy_train_state, critic_train_state, critic_target_params
    )
    return train_state, policy_network, critic_network


@partial(
    jax.jit,
    static_argnames=(
        "env_obj",
        "num_belief_particles",
        "num_planner_particles",
        "num_planner_steps",
        "random_actions",
    ),
)
def pomdp_rollout(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    train_state: JointTrainState,
    num_belief_particles: int,
    num_planner_particles: int,
    num_planner_steps: int,
    alpha: float,
    gamma: float,
    random_actions: bool,
) -> POMDPState:
    """Simulate trajectories with the DSMC policy."""

    def body(pomdp_state, key):
        pomdp_state = pomdp_step(
            rng_key=key,
            env_obj=env_obj,
            train_state=train_state,
            pomdp_state=pomdp_state,
            num_belief_particles=num_belief_particles,
            num_planner_particles=num_planner_particles,
            num_planner_steps=num_planner_steps,
            alpha=alpha,
            gamma=gamma,
            random_actions=random_actions,
        )
        return pomdp_state, pomdp_state

    init_key, scan_keys = custom_split(rng_key, env_obj.num_time_steps + 1)
    init_pomdp_state = pomdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        train_state=train_state,
        num_belief_particles=num_belief_particles,
        num_planner_particles=num_planner_particles,
        num_planner_steps=num_planner_steps,
        alpha=alpha,
        gamma=gamma,
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
