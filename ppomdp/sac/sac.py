import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from distrax import Chain, ScalarAffine
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array, random

import ppomdp.sac.replay_buffer as rb
from ppomdp.bijector import Tanh
from ppomdp.sac import pendulum
from ppomdp.sac.utils import (
    ActorNetwork,
    Environment,
    OuterState,
    SoftQNetwork,
    sample_and_log_prob,
)


class Args(NamedTuple):
    """Arguments for SAC."""

    seed: int = 1
    total_timesteps: int = 1000000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    print_freq: int = 50


def init(
    rng_key: PRNGKey,
    env: Environment,
    num_outer_particles: int,
    policy_state: TrainState,
) -> OuterState:
    key, state_key, action_key = random.split(rng_key, 3)
    states = env.prior_dist.sample(seed=state_key, sample_shape=(num_outer_particles,))
    actions, _, _ = policy_state.apply_fn(action_key, states, policy_state.params)
    rewards = jax.vmap(env.reward_fn, in_axes=(0, 0, None))(states, actions, 0)

    keys = random.split(key, num_outer_particles)
    next_states = jax.vmap(env.trans_model.sample)(keys, states, actions)
    time_steps = jnp.ones(num_outer_particles)
    dones = jnp.zeros(num_outer_particles)

    outer_state = OuterState(
        states, actions, next_states, rewards, time_steps, dones, rewards.copy()
    )
    return outer_state


@partial(jax.jit, static_argnums=1, donate_argnums=3)
def step(
    rng_key: PRNGKey,
    env: Environment,
    policy_state: TrainState,
    outer_state: OuterState,
) -> OuterState:
    num_particles = outer_state.states.shape[0]

    def true_fn(_outer_state: OuterState) -> OuterState:
        keys = random.split(rng_key, num_particles + 1)
        states = _outer_state.next_states
        actions, _, _ = policy_state.apply_fn(keys[0], states, policy_state.params)
        rewards = jax.vmap(env.reward_fn)(states, actions, _outer_state.time_steps)

        next_states = jax.vmap(env.trans_model.sample)(
            keys[1:], _outer_state.states, _outer_state.actions
        )
        time_steps = _outer_state.time_steps + 1
        dones = jax.lax.cond(
            time_steps[0] >= env.num_time_steps,
            lambda _: jnp.ones(num_particles),
            lambda _: jnp.zeros(num_particles),
            None,
        )
        eps_rewards = _outer_state.episodic_rewards + rewards

        return OuterState(
            states, actions, next_states, rewards, time_steps, dones, eps_rewards
        )

    # Restart from the first time step if the episode is done.
    outer_state = jax.lax.cond(
        outer_state.dones[0] == 0,
        true_fn,
        lambda _: init(rng_key, env, num_particles, policy_state),
        outer_state,
    )
    return outer_state


class JointTrainState(NamedTuple):
    policy_state: TrainState
    q_state: TrainState
    q_target_params: tuple[FrozenDict, FrozenDict]


def create_train_state(
    rng_key: PRNGKey,
    env: Environment,
    q_lr: float,
    policy_lr: float,
) -> JointTrainState:
    actor_network = ActorNetwork(env.action_dim)
    bijector = Chain([ScalarAffine(env.action_shift, env.action_scale), Tanh()])
    actor = partial(sample_and_log_prob, network=actor_network, bijector=bijector)
    qf1 = SoftQNetwork()
    qf2 = SoftQNetwork()

    key, state_key, action_key = random.split(rng_key, 3)
    init_states = random.uniform(state_key, (1, env.state_dim))
    init_actions = random.uniform(action_key, (1, env.action_dim))

    key, qf1_key, qf2_key = random.split(key, 3)
    qf1_params = qf1.init(qf1_key, init_states, init_actions)["params"]
    qf2_params = qf2.init(qf2_key, init_states, init_actions)["params"]
    q_params = (qf1_params, qf2_params)
    q_target_params = jax.tree.map(lambda x: x.copy(), q_params)

    key, policy_key = random.split(key)
    policy_params = actor_network.init(policy_key, init_states)["params"]

    q_optimizer = optax.adam(q_lr)
    policy_optimizer = optax.adam(policy_lr)

    @jax.jit
    def q_apply_fn(_params, states, actions):
        _qf1_params, _qf2_params = _params
        qf1_vals = qf1.apply({"params": _qf1_params}, states, actions)
        qf2_vals = qf2.apply({"params": _qf2_params}, states, actions)
        return jnp.squeeze(qf1_vals), jnp.squeeze(qf2_vals)

    q_train_state = TrainState.create(
        apply_fn=q_apply_fn, params=q_params, tx=q_optimizer
    )
    policy_train_state = TrainState.create(
        apply_fn=actor, params=policy_params, tx=policy_optimizer
    )
    return JointTrainState(policy_train_state, q_train_state, q_target_params)


@partial(jax.jit, donate_argnums=1)
def q_train_step(
    rng_key: PRNGKey,
    ts: JointTrainState,
    data: OuterState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    next_actions, next_log_probs, _ = ts.policy_state.apply_fn(
        rng_key, data.next_states, ts.policy_state.params
    )
    qf1_next_target, qf2_next_target = ts.q_state.apply_fn(
        ts.q_target_params, data.next_states, next_actions
    )
    min_qf_next_target = (
        jnp.minimum(qf1_next_target, qf2_next_target) - alpha * next_log_probs
    )
    next_q_value = data.rewards + (1 - data.dones) * gamma * min_qf_next_target

    def loss_fn(_params):
        qf1_a_values, qf2_a_values = ts.q_state.apply_fn(
            _params, data.states, data.actions
        )
        qf1_loss = optax.l2_loss(qf1_a_values, next_q_value)
        qf2_loss = optax.l2_loss(qf2_a_values, next_q_value)
        return jnp.sum(qf1_loss + qf2_loss)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(ts.q_state.params)
    new_q_state = ts.q_state.apply_gradients(grads=grads)
    ts = ts._replace(q_state=new_q_state)
    return ts, loss


@partial(jax.jit, donate_argnums=1)
def policy_train_step(
    rng_key: PRNGKey, ts: JointTrainState, data: OuterState, alpha: float
) -> tuple[JointTrainState, Array]:
    def loss_fn(_params):
        actions, log_probs, _ = ts.policy_state.apply_fn(rng_key, data.states, _params)
        qf1_pi, qf2_pi = ts.q_state.apply_fn(ts.q_state.params, data.states, actions)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
        return jnp.mean(alpha * log_probs - min_qf_pi)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(ts.policy_state.params)
    new_policy_state = ts.policy_state.apply_gradients(grads=grads)
    ts = ts._replace(policy_state=new_policy_state)
    return ts, loss


@partial(jax.jit, donate_argnums=0)
def q_target_update(ts: JointTrainState, tau: float) -> JointTrainState:
    updated_params = jax.tree.map(
        lambda p, tp: optax.incremental_update(p, tp, tau),
        ts.q_state.params,
        ts.q_target_params,
    )
    return ts._replace(q_target_params=updated_params)


if __name__ == "__main__":
    args = Args()

    env = pendulum.env
    key = random.PRNGKey(0)
    key, sub_key = random.split(key)
    ts = create_train_state(sub_key, env, args.q_lr, args.policy_lr)

    key, sub_key = random.split(key)
    outer_state = init(sub_key, env, args.batch_size, ts.policy_state)

    buffer_entry_prototype = jax.tree.map(lambda x: x[0], outer_state)
    buffer = rb.init(buffer_entry_prototype, args.buffer_size)
    buffer = rb.add_batch(buffer, outer_state)

    for global_step in range(1, args.total_timesteps):
        key, sub_key = random.split(key)
        # TODO: Before learning starts, sample action from a uniform distribution.
        outer_state = step(sub_key, env, ts.policy_state, outer_state)
        buffer = rb.add_batch(buffer, outer_state)

        if global_step > args.learning_starts:
            key, sub_key = random.split(key)
            data = rb.sample(sub_key, buffer, args.batch_size)

            # Update the Q function.
            key, q_key = random.split(key)
            ts, q_loss = q_train_step(q_key, ts, data, args.alpha, args.gamma)

            # Update the policy.
            if global_step % args.policy_frequency == 0:
                policy_loss_sum = 0.0
                for _ in range(args.policy_frequency):
                    key, policy_key = random.split(key)
                    ts, policy_loss = policy_train_step(
                        policy_key, ts, data, args.alpha
                    )
                    policy_loss_sum += float(policy_loss)

            # Update the target networks.
            if global_step % args.target_network_frequency == 0:
                ts = q_target_update(ts, args.tau)

        if outer_state.dones[0] == 1:
            print(
                f"Step: {global_step:6d} | Episodic reward: {outer_state.episodic_rewards.mean():6.2f}"
            )

    # Plot a sample trajectory once training is done.
