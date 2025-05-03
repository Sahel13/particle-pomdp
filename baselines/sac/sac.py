from functools import partial
from typing import NamedTuple, Dict

import jax
import optax

from jax import Array, random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from brax.training.replay_buffers import UniformSamplingQueue, ReplayBufferState
from distrax import Block

from ppomdp.core import PRNGKey
from ppomdp.envs.core import MDPEnv, MDPState
from ppomdp.bijector import Tanh

from baselines.common import JointTrainState, sample_random_actions
from baselines.sac.arch import PolicyNetwork, CriticNetwork
from baselines.sac.utils import policy_sample_and_log_prob

from copy import deepcopy


def mdp_init(
    rng_key: PRNGKey,
    env_obj: MDPEnv,
    policy_state: TrainState,
    random_actions: bool = False,
) -> MDPState:

    key, state_key, action_key = random.split(rng_key, 3)
    states = env_obj.init_dist.sample(seed=state_key, sample_shape=(env_obj.num_envs,))
    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    actions = sample_random_actions(action_key, env_obj) if random_actions \
        else policy_state.apply_fn(action_key, params=policy_state.params, state=states, time_idx=time_idxs)[0]

    rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)

    keys = random.split(key, env_obj.num_envs)
    next_states = jax.vmap(env_obj.trans_model.sample)(keys, states, actions)
    done_flags = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    return MDPState(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        total_rewards=rewards.copy(),
        time_idxs=time_idxs,
        done_flags=done_flags,
    )


@partial(
    jax.jit,
    static_argnames=("env_obj", "random_actions"),
    donate_argnames="mdp_state"
)
def mdp_step(
    rng_key: PRNGKey,
    env_obj: MDPEnv,
    policy_state: TrainState,
    mdp_state: MDPState,
    random_actions: bool = False,
) -> MDPState:

    def _true_fn(_mdp_state):
        key, action_key = random.split(rng_key, 2)
        states = _mdp_state.next_states
        time_idxs = _mdp_state.time_idxs + 1

        actions = sample_random_actions(action_key, env_obj) if random_actions \
            else policy_state.apply_fn(action_key, params=policy_state.params, state=states, time_idx=time_idxs)[0]

        state_keys = random.split(key, env_obj.num_envs)
        rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
        total_rewards = _mdp_state.total_rewards + rewards
        next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)
        done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(jnp.int32)

        return MDPState(
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            total_rewards=total_rewards,
            time_idxs=time_idxs,
            done_flags=done_flags
        )

    def _false_fn(_mdp_state):
        return mdp_init(
            rng_key=rng_key,
            env_obj=env_obj,
            policy_state=policy_state,
            random_actions=random_actions
        )

    return jax.lax.cond(
        jnp.all(mdp_state.done_flags == 0),
        _true_fn,
        _false_fn,
        mdp_state
    )


def critic_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    mdp_state: MDPState,
    alpha: float,
    gamma: float
) -> tuple[JointTrainState, Array]:
    next_actions, next_log_probs, _ = \
        train_state.policy_state.apply_fn(
            rng_key=rng_key,
            params=train_state.policy_state.params,
            state=mdp_state.next_states,
            time_idx=mdp_state.time_idxs,
        )
    _next_values = \
        train_state.critic_state.apply_fn(
            train_state.critic_target_params,
            mdp_state.next_states,
            next_actions,
            mdp_state.time_idxs + 1
        )
    next_values = jnp.min(_next_values, axis=-1) - alpha * next_log_probs
    target_values = mdp_state.rewards + (1 - mdp_state.done_flags) * gamma * next_values

    def critic_loss(params):
        _values = \
            train_state.critic_state.apply_fn(
                params,
                mdp_state.states,
                mdp_state.actions,
                mdp_state.time_idxs
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
    mdp_state: MDPState,
    alpha: float
) -> tuple[JointTrainState, Array]:
    def policy_loss(params):
        actions, log_probs, _ = \
            train_state.policy_state.apply_fn(
                rng_key=rng_key,
                params=params,
                state=mdp_state.states,
                time_idx=mdp_state.time_idxs
            )
        values = \
            train_state.critic_state.apply_fn(
                train_state.critic_state.params,
                mdp_state.states,
                actions,
                mdp_state.time_idxs,
            )
        min_values = jnp.min(values, axis=-1)
        return jnp.mean(alpha * log_probs - min_values)

    grad_fn = jax.value_and_grad(policy_loss)
    loss, grads = grad_fn(train_state.policy_state.params)
    _policy_state = train_state.policy_state.apply_gradients(grads=grads)
    train_state = train_state._replace(policy_state=_policy_state)
    return train_state, loss


def critic_target_update(
    train_state: JointTrainState,
    tau: float
) -> JointTrainState:
    updated_params = jax.tree.map(
        lambda param, target: tau * param + (1 - tau) * target,
        train_state.critic_state.params,
        train_state.critic_target_params,
    )
    return train_state._replace(critic_target_params=updated_params)


def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    mdp_state: MDPState,
    alpha: float,
    gamma: float,
    tau: float,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)

    # Update policy
    train_state, policy_loss = policy_train_step(
        rng_key=policy_key,
        train_state=train_state,
        mdp_state=mdp_state,
        alpha=alpha,
    )

    # Update critic
    train_state, critic_loss = critic_train_step(
        rng_key=critic_key,
        train_state=train_state,
        mdp_state=mdp_state,
        alpha=alpha,
        gamma=gamma,
    )

    # Update target critic
    train_state = critic_target_update(train_state, tau)
    return train_state, policy_loss, critic_loss


@partial(
    jax.jit,
    static_argnames=("env_obj", "buffer_obj", "num_steps", "alpha", "gamma", "tau"),
    donate_argnames=("buffer_state", "mdp_state", "train_state"),
)
def step_and_train(
    rng_key: PRNGKey,
    env_obj: MDPEnv,
    train_state: JointTrainState,
    mdp_state: MDPState,
    buffer_obj: UniformSamplingQueue,
    buffer_state: ReplayBufferState,
    num_steps: int,
    alpha: float,
    gamma: float,
    tau: float,
):

    def body(carry, key):
        _mdp_state, _buffer_state, _train_state = carry
        _step_key, _train_key = random.split(key)

        _mdp_state = mdp_step(
            rng_key=_step_key,
            env_obj=env_obj,
            policy_state=_train_state.policy_state,
            mdp_state=_mdp_state,
        )
        _buffer_state = buffer_obj.insert(_buffer_state, _mdp_state)
        _buffer_state, _mdp_state_sample = buffer_obj.sample(_buffer_state)

        _train_state, _, _ = \
            gradient_step(
                rng_key=_train_key,
                train_state=_train_state,
                mdp_state=_mdp_state_sample,
                alpha=alpha,
                gamma=gamma,
                tau=tau,
            )
        return (_mdp_state, _buffer_state, _train_state), None

    keys = random.split(rng_key, num_steps)
    (mdp_state, buffer_state, train_state), _ = \
        jax.lax.scan(body, (mdp_state, buffer_state, train_state), keys)

    return mdp_state, buffer_state, train_state


def create_train_state(
    rng_key: PRNGKey,
    env_obj: MDPEnv,
    policy_lr: float,
    critic_lr: float,
) -> JointTrainState:

    policy_network = PolicyNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        hidden_sizes=(256, 256),
        output_dim=env_obj.action_dim,
    )
    critic_networks = CriticNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        hidden_sizes=(256, 256),
        num_critics=2,
    )

    dummy_states = jnp.empty((1, env_obj.state_dim))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_time_idxs = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_states, dummy_time_idxs)["params"]
    critic_params = critic_networks.init(critic_key, dummy_states, dummy_actions, dummy_time_idxs)
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
        tx=optax.adam(policy_lr)
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_networks.apply,
        params=critic_params,
        tx=optax.adam(critic_lr)
    )
    return JointTrainState(
        policy_train_state,
        critic_train_state,
        critic_target_params
    )
