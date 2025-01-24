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
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import Array, random

from ppomdp import smc
from ppomdp.core import InnerState
from ppomdp.envs import pendulum
from ppomdp.envs.base import Environment
from ppomdp.policy import LSTM, reset_policy
from ppomdp.sac.sac import sample_random_actions
from ppomdp.sac.utils import QNetworks
from ppomdp.slac.utils import (
    OuterState,
    Transition,
    get_transition,
    sample_and_log_prob,
)
from ppomdp.utils import systematic_resampling


class Args(NamedTuple):
    """Arguments for SAC from cleanrl."""

    seed: int = 1
    total_timesteps: int = int(1e5)  # int(1e6)
    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 1e-4
    q_lr: float = 1e-3
    alpha: float = 0.2
    num_envs: int = 1
    num_particles: int = 2  # For the particle filter.


def pf_init(
    rng_key: chex.PRNGKey, env: Environment, observation: Array, num_particles: int
) -> InnerState:
    """Initialize the particle filter to track the belief state."""
    particles = env.prior_dist.sample(seed=rng_key, sample_shape=(num_particles,))
    log_weights = jax.vmap(env.obs_model.log_prob, (None, 0))(observation, particles)
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    dummy_resampling_indices = jnp.zeros(num_particles, dtype=jnp.int32)
    return InnerState(particles, log_weights, weights, dummy_resampling_indices)


def pf_step(
    rng_key: chex.PRNGKey,
    env: Environment,
    observation: Array,
    action: Array,
    belief_state: InnerState,
) -> InnerState:
    """Single step of the particle filter to track the belief state."""
    key, sub_key = random.split(rng_key)
    resampled_state = smc.resample_inner(sub_key, belief_state, systematic_resampling)
    key, sub_key = random.split(key)
    particles = smc.propagate_inner(
        sub_key, env.trans_model, resampled_state.particles, action
    )
    resampled_state = resampled_state._replace(particles=particles)
    belief_state = smc.reweight_inner(env.obs_model, resampled_state, observation)
    return belief_state


def _step_atom(
    rng_key,
    env,
    states,
    observations,
    carry,
    belief_states,
    policy_state,
    random_actions,
):
    num_envs = observations.shape[0]

    # Sample actions.
    key, sub_key = random.split(rng_key)
    if random_actions:
        actions = sample_random_actions(sub_key, env, num_envs)
        next_carry = carry
    else:
        next_carry, actions, _, _ = policy_state.apply_fn(
            sub_key,
            policy_state.params,
            carry,
            observations,
        )

    # Get the next states.
    keys = random.split(key, num_envs + 1)
    next_states = jax.vmap(env.trans_model.sample)(keys[1:], states, actions)

    # Sample observations.
    keys = random.split(keys[0], num_envs + 1)
    next_observations = jax.vmap(env.obs_model.sample)(keys[1:], next_states)

    # Update the belief states.
    keys = random.split(keys[0], num_envs)
    next_belief_states = jax.vmap(pf_step, (0, None, 0, 0, 0))(
        keys, env, next_observations, actions, belief_states
    )

    return actions, next_states, next_observations, next_carry, next_belief_states


def init(
    rng_key: chex.PRNGKey,
    env: Environment,
    policy_state: TrainState,
    policy_network: LSTM,
    num_envs: int,
    num_particles: int = 100,
    random_actions: bool = False,
) -> OuterState:
    """Initialize the outer state."""
    key, sub_key = random.split(rng_key)
    states = env.prior_dist.sample(seed=sub_key, sample_shape=(num_envs,))

    keys = random.split(key, num_envs + 1)
    observations = jax.vmap(env.obs_model.sample)(keys[1:], states)

    keys = random.split(keys[0], num_envs + 1)
    belief_states = jax.vmap(pf_init, (0, None, 0, None))(
        keys[1:], env, observations, num_particles
    )

    key, sub_key = random.split(keys[0])
    carry = reset_policy(num_envs, policy_network)
    actions, next_states, next_observations, next_carry, next_belief_states = (
        _step_atom(
            sub_key,
            env,
            states,
            observations,
            carry,
            belief_states,
            policy_state,
            random_actions,
        )
    )

    rewards = jax.vmap(env.reward_fn, (0, 0, None))(states, actions, 0)
    episodic_rewards = rewards.copy()
    time_steps = jnp.ones(num_envs)
    dones = jnp.zeros(num_envs)

    outer_state = OuterState(
        states,
        observations,
        carry,
        belief_states,
        actions,
        next_states,
        next_observations,
        next_carry,
        next_belief_states,
        rewards,
        episodic_rewards,
        time_steps,
        dones,
    )
    return outer_state


@partial(jax.jit, static_argnums=(1, 3, 5), donate_argnames="outer_state")
def step(
    rng_key: chex.PRNGKey,
    env: Environment,
    policy_state: TrainState,
    policy_network: LSTM,
    outer_state: OuterState,
    random_actions: bool = False,
) -> OuterState:
    num_envs = outer_state.observations.shape[0]
    num_particles = outer_state.belief_states.particles.shape[1]

    def true_fn(_outer_state: OuterState) -> OuterState:
        states = _outer_state.next_states
        observations = _outer_state.next_observations
        carry = _outer_state.next_carry
        belief_states = _outer_state.next_belief_states
        actions, next_states, next_observations, next_carry, next_belief_states = (
            _step_atom(
                rng_key,
                env,
                states,
                observations,
                carry,
                belief_states,
                policy_state,
                random_actions,
            )
        )

        rewards = jax.vmap(env.reward_fn)(
            _outer_state.states, actions, _outer_state.time_steps
        )
        episodic_rewards = _outer_state.episodic_rewards + rewards
        time_steps = _outer_state.time_steps + 1
        dones = jax.lax.select(
            time_steps[0] >= env.num_time_steps, jnp.ones(num_envs), jnp.zeros(num_envs)
        )

        outer_state = OuterState(
            states,
            observations,
            carry,
            belief_states,
            actions,
            next_states,
            next_observations,
            next_carry,
            next_belief_states,
            rewards,
            episodic_rewards,
            time_steps,
            dones,
        )
        return outer_state

    # Restart from the first time step if the episode is done.
    outer_state = jax.lax.cond(
        outer_state.dones[0] == 0,
        true_fn,
        lambda _: init(
            rng_key, env, policy_state, policy_network, num_envs, num_particles
        ),
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
) -> tuple[JointTrainState, LSTM]:
    policy_network = LSTM(
        dim=env.action_dim,
        feature_fn=lambda x: x,
        encoder_size=(256, 256),
        recurr_size=(64, 64),
        output_size=(256, 256),
        init_log_std=constant(jnp.log(1.0)),
    )
    q_networks = QNetworks(env.feature_fn)

    key, sub_key = random.split(rng_key)
    init_states = jnp.empty((1, env.state_dim))
    init_actions = jnp.empty((1, env.action_dim))
    q_params = q_networks.init(sub_key, init_states, init_actions)
    q_target_params = jax.tree.map(lambda x: x.copy(), q_params)

    key, policy_key = random.split(key)
    init_carry = reset_policy(env.num_envs, policy_network)
    init_observations = jnp.empty((1, env.obs_dim))
    policy_params = policy_network.init(policy_key, init_carry, init_observations)[
        "params"
    ]

    q_optimizer = optax.adam(q_lr)
    policy_optimizer = optax.adam(policy_lr)

    q_train_state = TrainState.create(
        apply_fn=q_networks.apply, params=q_params, tx=q_optimizer
    )

    policy_apply_fn = partial(
        sample_and_log_prob,
        network=policy_network,
        action_shift=env.action_scale,
        action_scale=env.action_shift,
    )
    policy_train_state = TrainState.create(
        apply_fn=policy_apply_fn, params=policy_params, tx=policy_optimizer
    )
    return (
        JointTrainState(policy_train_state, q_train_state, q_target_params),
        policy_network,
    )


def q_train_step(
    rng_key: chex.PRNGKey,
    ts: JointTrainState,
    data: Transition,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    _, next_actions, next_log_probs, _ = ts.policy_state.apply_fn(
        rng_key, ts.policy_state.params, data.next_carry, data.next_observations
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
    rng_key: chex.PRNGKey, ts: JointTrainState, data: Transition, alpha: float
) -> tuple[JointTrainState, Array]:
    def actor_loss(params):
        _, actions, log_probs, _ = ts.policy_state.apply_fn(
            rng_key, params, data.carry, data.observations
        )
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


def gradient_step(
    rng_key: chex.PRNGKey,
    ts: JointTrainState,
    data: Transition,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array, Array]:
    q_key, policy_key = random.split(rng_key, 2)
    ts, q_loss = q_train_step(q_key, ts, data, alpha, gamma)
    ts, policy_loss = policy_train_step(policy_key, ts, data, alpha)
    ts = q_target_update(ts, tau=0.005)
    return ts, q_loss, policy_loss


# TODO: Check if this actually needs to be jitted.
@partial(
    jax.jit,
    static_argnames=("env", "policy_network", "buffer"),
    donate_argnames=("buffer_state", "outer_state", "ts"),
)
def sim_and_train(rng_key, buffer_state, outer_state, ts, env, policy_network, buffer):
    def body_fn(carry, key):
        _buffer_state, _outer_state, _ts = carry
        step_key, trans_key, train_key = random.split(key, 3)
        next_outer_state = step(
            step_key,
            env,
            _ts.policy_state,
            policy_network,
            _outer_state,
        )
        transition = get_transition(trans_key, next_outer_state, 1)
        next_buffer_state = buffer.insert(_buffer_state, transition)

        # Training
        next_buffer_state, data = buffer.sample(next_buffer_state)
        next_ts, _, _ = gradient_step(train_key, _ts, data, args.alpha, args.gamma)
        return (next_buffer_state, next_outer_state, next_ts), None

    (buffer_state, outer_state, ts), _ = jax.lax.scan(
        body_fn,
        (buffer_state, outer_state, ts),
        random.split(rng_key, env.num_time_steps),
    )
    return buffer_state, outer_state, ts


if __name__ == "__main__":
    args = Args()

    env = pendulum.env
    key = random.key(args.seed)
    key, sub_key = random.split(key)
    ts, policy_network = create_train_state(sub_key, env, args.q_lr, args.policy_lr)

    key, init_key, trans_key = random.split(key, 3)
    outer_state = init(
        init_key,
        env,
        ts.policy_state,
        policy_network,
        args.num_envs,
        args.num_particles,
        True,
    )
    transition = get_transition(trans_key, outer_state, 1)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], transition)
    buffer = UniformSamplingQueue(
        args.buffer_size, buffer_entry_prototype, sample_batch_size=args.batch_size
    )
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer.sample_internal = jax.jit(buffer.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer.init(buffer_key)
    buffer_state = buffer.insert(buffer_state, transition)

    # Pre-fill the buffer with random actions.
    for global_step in range(args.learning_starts):
        key, key1, key2 = random.split(key, 3)
        outer_state = step(
            key1,
            env,
            ts.policy_state,
            policy_network,
            outer_state,
            True,
        )
        transition = get_transition(key2, outer_state, 1)
        buffer_state = buffer.insert(buffer_state, transition)
        if outer_state.dones[0] == 1:
            print(
                f"Step: {global_step:6d} | Episodic reward: {outer_state.episodic_rewards.mean():6.2f}"
            )

    # Ensure that training starts with a fresh episode.
    outer_state = outer_state._replace(dones=jnp.ones(args.num_envs))

    # Training loop - slightly faster training with `jax.lax.scan`.
    for global_step in range(
        args.learning_starts, args.total_timesteps, env.num_time_steps
    ):
        key, sub_key = random.split(key)
        buffer_state, outer_state, ts = sim_and_train(
            sub_key, buffer_state, outer_state, ts, env, policy_network, buffer
        )
        if global_step % 200 == 0:
            print(
                f"Step: {global_step:6d} | Episodic reward: {outer_state.episodic_rewards.mean():6.2f}"
            )

    # Evalute the learned policy.
    state_list, action_list = [], []
    key, sub_key = random.split(key)
    states = env.prior_dist.sample(seed=sub_key, sample_shape=(args.num_envs,))
    keys = random.split(key, args.num_envs + 1)
    observations = jax.vmap(env.obs_model.sample)(keys[1:], states)
    carry = reset_policy(args.num_envs, policy_network)
    state_list.append(states)

    for i in range(env.num_time_steps):
        key, sub_key = random.split(keys[0])
        carry, _, _, actions = ts.policy_state.apply_fn(
            sub_key, ts.policy_state.params, carry, observations
        )
        action_list.append(actions)
        keys = random.split(key, args.num_envs + 1)
        states = jax.vmap(env.trans_model.sample)(keys[1:], states, actions)
        state_list.append(states)
        keys = random.split(keys[0], args.num_envs + 1)
        observations = jax.vmap(env.obs_model.sample)(keys[1:], states)

    states = jnp.stack(state_list[:-1], axis=0)
    actions = jnp.stack(action_list, axis=0)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(states[:, 0, 0])
    axs[0].set_ylabel("Angle")

    axs[1].plot(states[:, 0, 1])
    axs[1].set_ylabel("Angular velocity")

    axs[2].plot(actions[:, 0, 0])
    axs[2].set_ylabel("Action")
    axs[2].set_xlabel("Time")
    plt.show()
