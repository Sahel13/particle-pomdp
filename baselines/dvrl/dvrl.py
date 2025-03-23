from functools import partial
from typing import NamedTuple, Dict

import jax
import optax

from jax import Array, random, numpy as jnp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from brax.training.replay_buffers import (
    UniformSamplingQueue,
    ReplayBufferState
)
from distrax import Block

from ppomdp.core import PRNGKey, InnerState
from ppomdp.utils import (
    propagate_inner,
    resample_inner,
    reweight_inner,
    systematic_resampling,
    custom_split
)
from ppomdp.bijector import Tanh

from ppomdp.envs.core import POMDPEnv, POMDPState
from baselines.dvrl.arch import PolicyNetwork, CriticNetwork
from baselines.dvrl.utils import sample_random_actions, policy_sample_and_log_prob

from copy import deepcopy


class DVRLConfig(NamedTuple):
    seed: int = 1
    num_particles: int = 64
    total_timesteps: int = int(1e5)
    buffer_size: int = int(1e5)
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


def pf_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    observation: Array,
    num_particles: int
) -> InnerState:
    """Initialize the particle filter to track the belief state."""
    particles = env_obj.prior_dist.sample(seed=rng_key, sample_shape=(num_particles,))
    log_weights = jax.vmap(env_obj.obs_model.log_prob, (None, 0))(observation, particles)
    logsum_weights = jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights - logsum_weights)
    dummy_resampling_indices = jnp.zeros(num_particles, dtype=jnp.int32)
    return InnerState(particles, log_weights, weights, dummy_resampling_indices)


def pf_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    belief: InnerState,
    observation: Array,
    action: Array,
) -> InnerState:
    """Single step of the particle filter to track the belief state."""
    key, sub_key = random.split(rng_key, 2)
    resampled_state = resample_inner(sub_key, belief, systematic_resampling)
    key, sub_key = random.split(key, 2)
    particles = propagate_inner(
        rng_key=sub_key,
        model=env_obj.trans_model,
        particles=resampled_state.particles,
        action=action
    )
    resampled_state = resampled_state._replace(particles=particles)
    return reweight_inner(env_obj.obs_model, resampled_state, observation)


def _step_atom(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    states: Array,
    beliefs: InnerState,
    policy_state: TrainState,
    random_actions: bool,
) -> tuple[Array, Array, InnerState, Array]:
    """Sample actions and get the next states and observations."""

    # Sample actions.
    key, action_key = random.split(rng_key)
    if random_actions:
        actions = sample_random_actions(action_key, env_obj)
    else:
        actions, _, _ = policy_state.apply_fn(
            rng_key=action_key,
            params=policy_state.params,
            particles=beliefs.particles,
            weights=beliefs.weights,
        )

    # Get the next states.
    key, state_keys = custom_split(key, env_obj.num_envs + 1)
    next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

    # Sample observations.
    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)

    # Update the belief states.
    key, pf_keys = custom_split(key, env_obj.num_envs + 1)
    next_beliefs = jax.vmap(pf_step, (0, None, 0, 0, 0))(
        pf_keys, env_obj, beliefs, next_observations, actions
    )

    return next_states, next_observations, next_beliefs, actions


def pomdp_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    num_particles: int,
    random_actions: bool = False,
) -> POMDPState:
    """Initialize the outer state."""
    key, prior_key = random.split(rng_key)
    states = env_obj.prior_dist.sample(seed=prior_key, sample_shape=(env_obj.num_envs,))

    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

    key, pf_keys = custom_split(key, env_obj.num_envs + 1)
    beliefs = jax.vmap(pf_init, (0, None, 0, None))(
        pf_keys, env_obj, observations, num_particles
    )

    key, step_key = random.split(key, 2)
    next_states, next_observations, next_beliefs, actions = \
        _step_atom(
            rng_key=step_key,
            env_obj=env_obj,
            states=states,
            beliefs=beliefs,
            policy_state=policy_state,
            random_actions=random_actions
    )

    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)
    rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
    done_flags = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    return POMDPState(
        states=states,
        carry=[],
        observations=observations,
        beliefs=beliefs,
        actions=actions,
        next_states=next_states,
        next_observations=next_observations,
        next_carry=[],
        next_beliefs=next_beliefs,
        rewards=rewards,
        total_rewards=rewards.copy(),
        time_idxs=time_idxs,
        done_flags=done_flags,
    )


@partial(jax.jit, static_argnums=(1, 4), donate_argnames="pomdp_state")
def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    pomdp_state: POMDPState,
    random_actions: bool = False,
) -> POMDPState:

    num_particles = pomdp_state.beliefs.particles.shape[1]

    def _true_fn(_pomdp_state):
        time_idxs = _pomdp_state.time_idxs + 1
        states = _pomdp_state.next_states
        observations = _pomdp_state.next_observations
        beliefs = _pomdp_state.next_beliefs

        next_states, next_observations, next_beliefs, actions = \
            _step_atom(
                rng_key=rng_key,
                env_obj=env_obj,
                states=states,
                beliefs=beliefs,
                policy_state=policy_state,
                random_actions=random_actions
        )

        rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
        total_rewards = _pomdp_state.total_rewards + rewards
        done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(jnp.int32)

        return POMDPState(
            states=states,
            carry=[],
            observations=observations,
            beliefs=beliefs,
            actions=actions,
            next_states=next_states,
            next_carry=[],
            next_observations=next_observations,
            next_beliefs=next_beliefs,
            rewards=rewards,
            total_rewards=total_rewards,
            time_idxs=time_idxs,
            done_flags=done_flags,
        )

    def _false_fn(_outer_state):
        return pomdp_init(rng_key, env_obj, policy_state, num_particles, random_actions)

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0), _true_fn, _false_fn, pomdp_state
    )


def create_train_state(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_lr: float,
    critic_lr: float,
    num_particles: int,
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:
    policy_log_std = jnp.ones(env_obj.action_dim)
    policy_network = PolicyNetwork(
        feature_fn=env_obj.feature_fn,
        recurr_size=64,
        hidden_sizes=(256, 256),
        output_dim=env_obj.action_dim,
        init_log_std=constant(policy_log_std),
    )
    critic_networks = CriticNetwork(
        feature_fn=env_obj.feature_fn,
        time_norm=env_obj.num_time_steps,
        recurr_size=64,
        hidden_sizes=(256, 256),
        num_critics=2
    )

    dummy_particles = jnp.empty((1, num_particles, env_obj.state_dim))
    dummy_weights = jnp.empty((1, num_particles))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_time = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_particles, dummy_weights)["params"]
    critic_params = critic_networks.init(critic_key, dummy_particles, dummy_weights, dummy_actions, dummy_time)
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
    train_state = JointTrainState(policy_train_state, critic_train_state, critic_target_params)
    return train_state, policy_network, critic_networks


def critic_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    next_actions, next_log_probs, _ = \
        train_state.policy_state.apply_fn(
            rng_key=rng_key,
            params=train_state.policy_state.params,
            particles=pomdp_state.next_beliefs.particles,
            weights=pomdp_state.next_beliefs.weights
        )
    _next_values = \
        train_state.critic_state.apply_fn(
            train_state.critic_target_params,
            pomdp_state.next_beliefs.particles,
            pomdp_state.next_beliefs.weights,
            next_actions,
            pomdp_state.time_idxs + 1
        )
    next_values = jnp.min(_next_values, axis=-1) - alpha * next_log_probs
    target_values = pomdp_state.rewards + (1 - pomdp_state.done_flags) * gamma * next_values

    def critic_loss(params):
        _values = \
            train_state.critic_state.apply_fn(
                params,
                pomdp_state.beliefs.particles,
                pomdp_state.beliefs.weights,
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
    alpha: float
) -> tuple[JointTrainState, Array]:
    def actor_loss(params):
        actions, log_probs, _ = \
            train_state.policy_state.apply_fn(
                rng_key=rng_key,
                params=params,
                particles=pomdp_state.next_beliefs.particles,
                weights=pomdp_state.next_beliefs.weights
            )
        values = \
            train_state.critic_state.apply_fn(
                train_state.critic_state.params,
                pomdp_state.beliefs.particles,
                pomdp_state.beliefs.weights,
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


@jax.jit
def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    pomdp_state: POMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)
    train_state, critic_loss = critic_train_step(critic_key, train_state, pomdp_state, alpha, gamma)
    train_state, policy_loss = policy_train_step(policy_key, train_state, pomdp_state, alpha)
    train_state = critic_target_update(train_state, tau=0.005)
    return train_state, policy_loss, critic_loss


@partial(
    jax.jit,
    static_argnames=("env_obj", "buffer_obj", "num_steps", "alpha", "gamma"),
    donate_argnames=("buffer_state", "pomdp_state", "train_state"),
)
def step_and_train(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    pomdp_state: POMDPState,
    buffer_obj: UniformSamplingQueue,
    buffer_state: ReplayBufferState,
    train_state: JointTrainState,
    num_steps: int,
    alpha: float,
    gamma: float,
):
    def body(carry, key):
        _pomdp_state, _buffer_state, _train_state = carry
        _step_key, _train_key = random.split(key)

        _pomdp_state = pomdp_step(
            rng_key=_step_key,
            env_obj=env_obj,
            pomdp_state=_pomdp_state,
            policy_state=_train_state.policy_state,
        )
        _buffer_state = buffer_obj.insert(_buffer_state, _pomdp_state)
        _buffer_state, _pomdp_state_sample = buffer_obj.sample(_buffer_state)

        _train_state, _, _ = \
            gradient_step(
                rng_key=_train_key,
                train_state=_train_state,
                pomdp_state=_pomdp_state_sample,
                alpha=alpha,
                gamma=gamma
            )
        return (_pomdp_state, _buffer_state, _train_state), None

    keys = random.split(rng_key, num_steps)
    (pomdp_state, buffer_state, train_state), _ = jax.lax.scan(
        body, (pomdp_state, buffer_state, train_state), keys
    )
    return pomdp_state, buffer_state, train_state


# if __name__ == "__main__":
#     args = Args()
#     cmd_args = common.get_cmd_args()
#     env = common.get_env(cmd_args.env)
#
#     key = random.key(args.seed)
#     key, sub_key = random.split(key)
#     ts = create_train_state(sub_key, env, args.num_particles, args.q_lr, args.policy_lr)
#
#     key, init_key = random.split(key)
#     outer_state = init(
#         init_key,
#         env,
#         ts.policy_state,
#         env.num_envs,
#         args.num_particles,
#         True,
#     )
#     transition = get_transition(outer_state)
#
#     # Set up the replay buffer from Brax.
#     buffer_entry_prototype = jax.tree.map(lambda x: x[0], transition)
#     buffer = UniformSamplingQueue(
#         args.buffer_size, buffer_entry_prototype, sample_batch_size=args.batch_size
#     )
#     buffer.insert_internal = jax.jit(buffer.insert_internal)
#     buffer.sample_internal = jax.jit(buffer.sample_internal)
#
#     # Initialize the buffer state.
#     key, buffer_key = random.split(key)
#     buffer_state = buffer.init(buffer_key)
#     buffer_state = buffer.insert(buffer_state, transition)
#
#     # Training loop.
#     for global_step in range(1, args.total_timesteps):
#         key, sub_key = random.split(key)
#         if global_step < args.learning_starts:
#             outer_state = step(sub_key, env, ts.policy_state, outer_state, True)
#         else:
#             outer_state = step(sub_key, env, ts.policy_state, outer_state, False)
#         transition = get_transition(outer_state)
#         buffer_state = buffer.insert(buffer_state, transition)
#
#         if global_step >= args.learning_starts:
#             key, sub_key = random.split(key)
#             buffer_state, data = buffer.sample(buffer_state)
#             ts, _, _ = gradient_step(sub_key, ts, data, args.alpha, args.gamma)
#
#         if outer_state.dones[0] == 1:
#             print(
#                 f"Step: {global_step:7d} | "
#                 + f"Episodic reward: {outer_state.episodic_rewards.mean():10.2f} | "
#                 + f"Policy log std: {ts.policy_state.params['log_std'][0]:6.2f}"
#                 + f"{outer_state.time_steps[0]}"
#             )
#
#     # Evaluate the learned policy.
#     key, obs_key, belief_key = random.split(key)
#     state = env.prior_dist.mean()
#     observation = env.obs_model.sample(obs_key, state)
#     belief_state = pf_init(belief_key, env, observation, args.num_particles)
#
#     def body(carry, rng_key):
#         state, belief_state = carry
#         action_key, state_key, obs_key, pf_key = random.split(rng_key, 4)
#         _, _, action = ts.policy_state.apply_fn(
#             action_key,
#             ts.policy_state.params,
#             belief_state.particles,
#             belief_state.weights,
#         )
#         state = env.trans_model.sample(state_key, state, action)
#         observation = env.obs_model.sample(obs_key, state)
#         belief_state = pf_step(pf_key, env, observation, action, belief_state)
#         return (state, belief_state), (state, action)
#
#     _, (states, actions) = jax.lax.scan(
#         body, (state, belief_state), random.split(key, env.num_time_steps)
#     )
#     states = jnp.concatenate([state[None, ...], states], axis=0)
#     common.plot_trajectory(cmd_args.env, states, actions)
#
