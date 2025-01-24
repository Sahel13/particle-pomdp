import os

# If running on Wong, specify the GPU to use (0-4).
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from functools import partial
from typing import Dict, NamedTuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from brax.training.replay_buffers import UniformSamplingQueue
from flax.training.train_state import TrainState
from jax import Array, random

from ppomdp.envs import Environment, PendulumEnv
from ppomdp.sac.utils import (
    ActorNetwork,
    OuterState,
    QNetworks,
    sample_and_log_prob,
)


class Args(NamedTuple):
    """Arguments for SAC from cleanrl."""

    seed: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    gamma: float = 0.995
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    alpha: float = 0.2


def sample_random_actions(
    rng_key: chex.PRNGKey, env: Environment, batch_size: int
) -> Array:
    rand_uniforms = random.uniform(
        rng_key, (batch_size, env.action_dim), minval=-1.0, maxval=1.0
    )
    return (rand_uniforms * env.action_scale) + env.action_shift


def init(
    rng_key: chex.PRNGKey,
    env: Environment,
    num_outer_particles: int,
    policy_state: TrainState,
    random_actions: bool = False,
) -> OuterState:
    """Initialize the outer state."""
    key, state_key, action_key = random.split(rng_key, 3)
    states = env.prior_dist.sample(seed=state_key, sample_shape=(num_outer_particles,))
    if random_actions:
        actions = sample_random_actions(action_key, env, num_outer_particles)
    else:
        actions, _, _ = policy_state.apply_fn(action_key, policy_state.params, states)
    keys = random.split(key, num_outer_particles)
    next_states = jax.vmap(env.trans_model.sample)(keys, states, actions)
    rewards = jax.vmap(env.reward_fn, in_axes=(0, 0, None))(next_states, actions, 1)
    time_steps = jnp.ones(num_outer_particles)
    dones = jnp.zeros(num_outer_particles)

    outer_state = OuterState(
        states, actions, next_states, rewards, time_steps, dones, rewards.copy()
    )
    return outer_state


@partial(jax.jit, static_argnums=(1, 4), donate_argnums=3)
def step(
    rng_key: chex.PRNGKey,
    env: Environment,
    policy_state: TrainState,
    outer_state: OuterState,
    random_actions: bool = False,
) -> OuterState:
    num_particles = outer_state.states.shape[0]

    def true_fn(_outer_state: OuterState) -> OuterState:
        keys = random.split(rng_key, num_particles + 1)
        states = _outer_state.next_states
        if random_actions:
            actions = sample_random_actions(keys[0], env, num_particles)
        else:
            actions, _, _ = policy_state.apply_fn(keys[0], policy_state.params, states)
        next_states = jax.vmap(env.trans_model.sample)(keys[1:], states, actions)
        time_steps = _outer_state.time_steps + 1
        dones = jax.lax.select(
            time_steps[0] == env.num_time_steps,
            jnp.ones(num_particles),
            jnp.zeros(num_particles),
        )
        rewards = jax.vmap(env.reward_fn)(next_states, actions, time_steps)
        eps_rewards = _outer_state.episodic_rewards + rewards

        return OuterState(
            states, actions, next_states, rewards, time_steps, dones, eps_rewards
        )

    # Restart from the first time step if the episode is done.
    outer_state = jax.lax.cond(
        outer_state.dones[0] == 0,
        true_fn,
        lambda _: init(rng_key, env, num_particles, policy_state, random_actions),
        outer_state,
    )
    return outer_state


class JointTrainState(NamedTuple):
    policy_state: TrainState
    q_state: TrainState
    q_target_params: Dict


def create_train_state(
    rng_key: chex.PRNGKey,
    env: Environment,
    q_lr: float,
    policy_lr: float,
) -> JointTrainState:
    actor_network = ActorNetwork(env.action_dim, jnp.array(1.0), env.feature_fn)
    q_networks = QNetworks(env.feature_fn)

    q_key, policy_key = random.split(rng_key)
    init_states = jnp.empty((1, env.state_dim))
    init_actions = jnp.empty((1, env.action_dim))
    q_params = q_networks.init(q_key, init_states, init_actions)
    q_target_params = jax.tree.map(lambda x: x.copy(), q_params)

    policy_params = actor_network.init(policy_key, init_states)["params"]

    q_optimizer = optax.adam(q_lr)
    policy_optimizer = optax.adam(policy_lr)

    q_train_state = TrainState.create(
        apply_fn=q_networks.apply, params=q_params, tx=q_optimizer
    )

    actor_apply_fn = partial(
        sample_and_log_prob,
        network=actor_network,
        action_shift=env.action_scale,
        action_scale=env.action_shift,
    )
    policy_train_state = TrainState.create(
        apply_fn=actor_apply_fn, params=policy_params, tx=policy_optimizer
    )
    return JointTrainState(policy_train_state, q_train_state, q_target_params)


def q_train_step(
    rng_key: chex.PRNGKey,
    ts: JointTrainState,
    data: OuterState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    next_actions, next_log_probs, _ = ts.policy_state.apply_fn(
        rng_key, ts.policy_state.params, data.next_states
    )
    next_q = ts.q_state.apply_fn(ts.q_target_params, data.next_states, next_actions)
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_probs
    target_q = data.rewards + (1 - data.dones) * gamma * next_v

    def critic_loss(params):
        q_old = ts.q_state.apply_fn(params, data.states, data.actions)
        q_error = q_old - jnp.expand_dims(target_q, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    grad_fn = jax.value_and_grad(critic_loss)
    loss, grads = grad_fn(ts.q_state.params)
    new_q_state = ts.q_state.apply_gradients(grads=grads)
    ts = ts._replace(q_state=new_q_state)
    return ts, loss


def policy_train_step(
    rng_key: chex.PRNGKey, ts: JointTrainState, data: OuterState, alpha: float
) -> tuple[JointTrainState, Array]:
    def actor_loss(params):
        actions, log_probs, _ = ts.policy_state.apply_fn(rng_key, params, data.states)
        q_vals = ts.q_state.apply_fn(ts.q_state.params, data.states, actions)
        min_qf_pi = jnp.min(q_vals, axis=-1)
        return jnp.mean(alpha * log_probs - min_qf_pi)

    grad_fn = jax.value_and_grad(actor_loss)
    loss, grads = grad_fn(ts.policy_state.params)
    new_policy_state = ts.policy_state.apply_gradients(grads=grads)
    ts = ts._replace(policy_state=new_policy_state)
    return ts, loss


def q_target_update(ts: JointTrainState, tau: float) -> JointTrainState:
    updated_params = jax.tree.map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        ts.q_state.params,
        ts.q_target_params,
    )
    return ts._replace(q_target_params=updated_params)


@partial(jax.jit, donate_argnums=1)
def gradient_step(
    rng_key: chex.PRNGKey,
    ts: JointTrainState,
    data: OuterState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array, Array]:
    q_key, policy_key = random.split(rng_key, 2)
    ts, q_loss = q_train_step(q_key, ts, data, alpha, gamma)
    ts, policy_loss = policy_train_step(policy_key, ts, data, alpha)
    ts = q_target_update(ts, tau=0.005)
    return ts, q_loss, policy_loss


if __name__ == "__main__":
    args = Args()

    env = PendulumEnv
    key = random.key(args.seed)
    key, sub_key = random.split(key)
    ts = create_train_state(sub_key, env, args.q_lr, args.policy_lr)

    key, sub_key = random.split(key)
    outer_state = init(sub_key, env, env.num_envs, ts.policy_state, True)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], outer_state)
    buffer = UniformSamplingQueue(
        args.buffer_size, buffer_entry_prototype, sample_batch_size=args.batch_size
    )
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer.sample_internal = jax.jit(buffer.sample_internal)

    key, sub_key = random.split(key)
    buffer_state = buffer.init(sub_key)
    buffer_state = buffer.insert(buffer_state, outer_state)

    for global_step in range(1, args.total_timesteps):
        key, sub_key = random.split(key)
        if global_step < args.learning_starts:
            outer_state = step(sub_key, env, ts.policy_state, outer_state, True)
        else:
            outer_state = step(sub_key, env, ts.policy_state, outer_state)
        buffer_state = buffer.insert(buffer_state, outer_state)

        if global_step > args.learning_starts:
            buffer_state, data = buffer.sample(buffer_state)
            key, sub_key = random.split(key)
            ts, q_loss, policy_loss = gradient_step(
                sub_key, ts, data, args.alpha, args.gamma
            )

        if outer_state.dones[0] == 1:
            print(
                f"Step: {global_step:6d} | "
                + f"Episodic reward: {outer_state.episodic_rewards.mean():10.2f}"
            )

    # Evaluate the learned policy.
    key, state_key = random.split(key)
    state = env.prior_dist.sample(seed=state_key)

    def body_fn(state, rng_key):
        action_key, state_key = random.split(rng_key)
        _, _, action = ts.policy_state.apply_fn(
            action_key, ts.policy_state.params, state
        )
        state = env.trans_model.sample(state_key, state, action)
        return state, (state, action)

    _, (states, actions) = jax.lax.scan(
        body_fn, state, random.split(key, env.num_time_steps)
    )
    states = jnp.concatenate([state[None, ...], states], axis=0)

    # For the pendulum environment.
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("Simulated trajectory")

    axs[0].plot(states[:, 0])
    axs[0].set_ylabel("Angle")
    axs[0].grid(True)

    axs[1].plot(states[:, 1])
    axs[1].set_ylabel("Angular velocity")
    axs[1].grid(True)

    axs[2].plot(actions[:, 0])
    axs[2].set_ylabel("Action")
    axs[2].set_xlabel("Time")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
