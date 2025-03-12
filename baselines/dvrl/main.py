from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from brax.training.replay_buffers import UniformSamplingQueue
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from jax import Array, random

from baselines import common
from baselines.dvrl.utils import (
    OuterState,
    PolicyNetwork,
    QNetworks,
    Transition,
    get_transition,
    sample_and_log_prob,
)
from baselines.sac import JointTrainState, q_target_update, sample_random_actions
from baselines.slac.slac import pf_init, pf_step
from ppomdp.core import InnerState
from ppomdp.envs import Environment


class Args(NamedTuple):
    seed: int = 1
    total_timesteps: int = int(1e4)
    buffer_size: int = int(1e5)
    gamma: float = 0.995
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 1e-4
    q_lr: float = 1e-3
    alpha: float = 0.2
    num_particles: int = 64


def _step_atom(
    rng_key: chex.PRNGKey,
    env: Environment,
    states: Array,
    belief_states: InnerState,
    policy_state: TrainState,
    random_actions: bool,
) -> tuple[Array, Array, InnerState]:
    """Sample actions and get the next states and observations."""
    num_envs = states.shape[0]

    # Sample actions.
    key, action_key = random.split(rng_key)
    if random_actions:
        actions = sample_random_actions(action_key, env, num_envs)
    else:
        actions, _, _ = policy_state.apply_fn(
            action_key,
            policy_state.params,
            belief_states.particles,
            belief_states.weights,
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

    return actions, next_states, next_belief_states


def init(
    rng_key: chex.PRNGKey,
    env: Environment,
    policy_state: TrainState,
    num_envs: int,
    num_particles: int,
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

    actions, next_states, next_belief_states = _step_atom(
        keys[0], env, states, belief_states, policy_state, random_actions
    )
    time_steps = jnp.zeros(num_envs, dtype=jnp.int32)
    rewards = jax.vmap(env.reward_fn)(states, actions, time_steps)
    episodic_rewards = rewards.copy()
    dones = jnp.zeros(num_envs)

    return OuterState(
        states,
        belief_states,
        actions,
        next_states,
        next_belief_states,
        rewards,
        episodic_rewards,
        time_steps,
        dones,
    )


@partial(jax.jit, static_argnums=(1, 4), donate_argnames="outer_state")
def step(
    rng_key: chex.PRNGKey,
    env: Environment,
    policy_state: TrainState,
    outer_state: OuterState,
    random_actions: bool = False,
) -> OuterState:
    num_envs = outer_state.states.shape[0]
    num_particles = outer_state.belief_states.particles.shape[1]

    def true_fn(_outer_state: OuterState) -> OuterState:
        states = _outer_state.next_states
        belief_states = _outer_state.next_belief_states
        actions, next_states, next_belief_states = _step_atom(
            rng_key, env, states, belief_states, policy_state, random_actions
        )
        time_steps = _outer_state.time_steps + 1
        rewards = jax.vmap(env.reward_fn)(states, actions, time_steps)
        episodic_rewards = _outer_state.episodic_rewards + rewards
        dones = jax.lax.select(
            time_steps[0] == env.num_time_steps, jnp.ones(num_envs), jnp.zeros(num_envs)
        )

        outer_state = OuterState(
            states,
            belief_states,
            actions,
            next_states,
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
        lambda _: init(rng_key, env, policy_state, num_envs, num_particles),
        outer_state,
    )
    return outer_state


def create_train_state(
    rng_key: chex.PRNGKey,
    env: Environment,
    num_particles: int,
    q_lr: float,
    policy_lr: float,
) -> JointTrainState:
    policy_network = PolicyNetwork(
        action_dim=env.action_dim,
        feature_fn=env.feature_fn,
        recur_size=64,
        init_log_std=constant(jnp.log(2.0)),
    )
    q_networks = QNetworks(env.feature_fn, env.num_time_steps, 64)

    q_key, policy_key = random.split(rng_key)
    particles = jnp.empty((1, num_particles, env.state_dim))
    weights = jnp.empty((1, num_particles))
    actions = jnp.empty((1, env.action_dim))
    time = jnp.empty((1,))
    q_params = q_networks.init(q_key, particles, weights, actions, time)
    q_target_params = jax.tree.map(lambda x: x.copy(), q_params)
    policy_params = policy_network.init(policy_key, particles, weights)["params"]

    q_optimizer = optax.adam(q_lr)
    policy_optimizer = optax.adam(policy_lr)

    q_train_state = TrainState.create(
        apply_fn=q_networks.apply, params=q_params, tx=q_optimizer
    )

    policy_apply_fn = partial(
        sample_and_log_prob,
        network=policy_network,
        action_scale=env.action_scale,
        action_shift=env.action_shift,
    )
    policy_train_state = TrainState.create(
        apply_fn=policy_apply_fn, params=policy_params, tx=policy_optimizer
    )
    return JointTrainState(policy_train_state, q_train_state, q_target_params)


def q_train_step(
    rng_key: chex.PRNGKey,
    ts: JointTrainState,
    data: Transition,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    next_actions, next_log_probs, _ = ts.policy_state.apply_fn(
        rng_key, ts.policy_state.params, *data.next_belief_states
    )
    next_q = ts.q_state.apply_fn(
        ts.q_target_params, *data.next_belief_states, next_actions, data.time_steps + 1
    )
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_probs
    target_q = data.rewards + (1 - data.dones) * gamma * next_v

    def critic_loss(params):
        q_old = ts.q_state.apply_fn(
            params, *data.belief_states, data.actions, data.time_steps
        )
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
        actions, log_probs, _ = ts.policy_state.apply_fn(
            rng_key, params, *data.belief_states
        )
        q_vals = ts.q_state.apply_fn(
            ts.q_state.params, *data.belief_states, actions, data.time_steps
        )
        min_qf_pi = jnp.min(q_vals, axis=-1)
        return jnp.mean(alpha * log_probs - min_qf_pi)

    grad_fn = jax.value_and_grad(actor_loss)
    loss, grads = grad_fn(ts.policy_state.params)
    new_policy_state = ts.policy_state.apply_gradients(grads=grads)
    ts = ts._replace(policy_state=new_policy_state)
    return ts, loss


@jax.jit
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
    ts = q_target_update(ts)
    return ts, q_loss, policy_loss


if __name__ == "__main__":
    args = Args()
    cmd_args = common.get_cmd_args()
    env = common.get_env(cmd_args.env)

    key = random.key(args.seed)
    key, sub_key = random.split(key)
    ts = create_train_state(sub_key, env, args.num_particles, args.q_lr, args.policy_lr)

    key, init_key = random.split(key)
    outer_state = init(
        init_key,
        env,
        ts.policy_state,
        env.num_envs,
        args.num_particles,
        True,
    )
    transition = get_transition(outer_state)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], transition)
    buffer = UniformSamplingQueue(
        args.buffer_size, buffer_entry_prototype, sample_batch_size=args.batch_size
    )
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer.sample_internal = jax.jit(buffer.sample_internal)

    # Initialize the buffer state.
    key, buffer_key = random.split(key)
    buffer_state = buffer.init(buffer_key)
    buffer_state = buffer.insert(buffer_state, transition)

    # Training loop.
    for global_step in range(1, args.total_timesteps):
        key, sub_key = random.split(key)
        if global_step < args.learning_starts:
            outer_state = step(sub_key, env, ts.policy_state, outer_state, True)
        else:
            outer_state = step(sub_key, env, ts.policy_state, outer_state, False)
        transition = get_transition(outer_state)
        buffer_state = buffer.insert(buffer_state, transition)

        if global_step >= args.learning_starts:
            key, sub_key = random.split(key)
            buffer_state, data = buffer.sample(buffer_state)
            ts, _, _ = gradient_step(sub_key, ts, data, args.alpha, args.gamma)

        if outer_state.dones[0] == 1:
            print(
                f"Step: {global_step:7d} | "
                + f"Episodic reward: {outer_state.episodic_rewards.mean():10.2f} | "
                + f"Policy log std: {ts.policy_state.params['log_std'][0]:6.2f}"
                + f"{outer_state.time_steps[0]}"
            )

    # Evaluate the learned policy.
    key, obs_key, belief_key = random.split(key)
    state = env.prior_dist.mean()
    observation = env.obs_model.sample(obs_key, state)
    belief_state = pf_init(belief_key, env, observation, args.num_particles)

    def body(carry, rng_key):
        state, belief_state = carry
        action_key, state_key, obs_key, pf_key = random.split(rng_key, 4)
        _, _, action = ts.policy_state.apply_fn(
            action_key,
            ts.policy_state.params,
            belief_state.particles,
            belief_state.weights,
        )
        state = env.trans_model.sample(state_key, state, action)
        observation = env.obs_model.sample(obs_key, state)
        belief_state = pf_step(pf_key, env, observation, action, belief_state)
        return (state, belief_state), (state, action)

    _, (states, actions) = jax.lax.scan(
        body, (state, belief_state), random.split(key, env.num_time_steps)
    )
    states = jnp.concatenate([state[None, ...], states], axis=0)
    common.plot_trajectory(cmd_args.env, states, actions)

