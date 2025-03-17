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

from ppomdp.core import PRNGKey, Carry, InnerState
from ppomdp.utils import (
    propagate_inner,
    resample_inner,
    reweight_inner,
    systematic_resampling,
    custom_split
)
from ppomdp.bijector import Tanh

from ppomdp.envs.core import POMDPEnv, POMDPState, QMDPState
from baselines.slac.arch import PolicyNetwork, CriticNetwork, GRUEncoder, MLPDecoder
from baselines.slac.utils import get_qmdp_state, sample_random_actions, policy_sample_and_log_prob

from copy import deepcopy


class SLACConfig(NamedTuple):
    seed: int = 1
    num_particles: int = 512
    total_timesteps: int = int(1e6)
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
    carry: list[Carry],
    observations: Array,
    beliefs: InnerState,
    policy_state: TrainState,
    random_actions: bool,
) -> tuple[Array, list[Carry], Array, InnerState, Array]:
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
    key, pf_keys = custom_split(key, env_obj.num_envs + 1)
    next_beliefs = jax.vmap(pf_step, (0, None, 0, 0, 0))(
        pf_keys, env_obj, beliefs, next_observations, actions
    )

    return next_states, next_carry, next_observations, next_beliefs, actions


def pomdp_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    num_particles: int,
    random_actions: bool = False,
) -> POMDPState:
    """Initialize the env state."""
    key, prior_key = random.split(rng_key, 2)
    states = env_obj.prior_dist.sample(seed=prior_key, sample_shape=(env_obj.num_envs,))

    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, states)

    key, pf_keys = custom_split(key, env_obj.num_envs + 1)
    beliefs = jax.vmap(pf_init, (0, None, 0, None))(pf_keys, env_obj, observations, num_particles)

    carry = policy_network.reset(env_obj.num_envs)

    key, step_key = random.split(key, 2)
    next_states, next_carry, next_observations, next_beliefs, actions = \
        _step_atom(
            rng_key=step_key,
            env_obj=env_obj,
            states=states,
            carry=carry,
            observations=observations,
            beliefs=beliefs,
            policy_state=policy_state,
            random_actions=random_actions,
        )

    time_idxs = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)
    rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
    done_flags = jnp.zeros(env_obj.num_envs, dtype=jnp.int32)

    return POMDPState(
        states=states,
        carry=carry,
        observations=observations,
        beliefs=beliefs,
        actions=actions,
        next_states=next_states,
        next_observations=next_observations,
        next_carry=next_carry,
        next_beliefs=next_beliefs,
        rewards=rewards,
        total_rewards=rewards.copy(),
        time_idxs=time_idxs,
        done_flags=done_flags,
    )


@partial(jax.jit, static_argnums=(1, 4, 5, 6), donate_argnames="pomdp_state")
def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    pomdp_state: POMDPState,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    num_particles: int,
    random_actions: bool = False,
) -> POMDPState:

    def _true_fn(_pomdp_state):
        time_idxs = _pomdp_state.time_idxs + 1
        states = _pomdp_state.next_states
        carry = _pomdp_state.next_carry
        observations = _pomdp_state.next_observations
        beliefs = _pomdp_state.next_beliefs

        next_states, next_carry, next_observations, next_beliefs, actions = \
            _step_atom(
                rng_key=rng_key,
                env_obj=env_obj,
                states=states,
                carry=carry,
                observations=observations,
                beliefs=beliefs,
                policy_state=policy_state,
                random_actions=random_actions,
            )

        rewards = jax.vmap(env_obj.reward_fn)(states, actions, time_idxs)
        total_rewards = _pomdp_state.total_rewards + rewards
        done_flags = jnp.where(time_idxs == env_obj.num_time_steps, 1, 0).astype(jnp.int32)

        return POMDPState(
            states=states,
            carry=carry,
            observations=observations,
            beliefs=beliefs,
            actions=actions,
            next_states=next_states,
            next_carry=next_carry,
            next_observations=next_observations,
            next_beliefs=next_beliefs,
            rewards=rewards,
            total_rewards=total_rewards,
            time_idxs=time_idxs,
            done_flags=done_flags,
        )

    def _false_fn(_outer_state):
        return pomdp_init(rng_key, env_obj, policy_state, policy_network, num_particles, random_actions)

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0), _true_fn, _false_fn, pomdp_state
    )


def create_train_state(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_lr: float,
    critic_lr: float,
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:
    policy_log_std = jnp.ones(env_obj.action_dim)
    policy_encoder = GRUEncoder(
        feature_fn=lambda x: x,
        encoder_size=(256, 256),
        recurr_size=(64, 64),
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
        layer_sizes=(256, 256),
        num_critics=2,
    )

    dummy_states = jnp.empty((1, env_obj.state_dim))
    dummy_actions = jnp.empty((1, env_obj.action_dim))
    dummy_carry = policy_network.reset(env_obj.num_envs)
    dummy_observations = jnp.empty((1, env_obj.obs_dim))
    dummy_time = jnp.empty((1,), dtype=jnp.int32)

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_carry, dummy_observations)["params"]
    critic_params = critic_networks.init(critic_key, dummy_states, dummy_actions, dummy_time)
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
    qmdp_state: QMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:
    _, next_actions, next_log_probs, _ = \
        train_state.policy_state.apply_fn(
            rng_key=rng_key,
            params=train_state.policy_state.params,
            carry=qmdp_state.next_carry,
            observation=qmdp_state.next_observations
        )
    _next_values = \
        train_state.critic_state.apply_fn(
            train_state.critic_target_params,
            qmdp_state.next_states,
            next_actions,
            qmdp_state.time_idxs + 1
        )
    next_values = jnp.min(_next_values, axis=-1) - alpha * next_log_probs
    target_values = qmdp_state.rewards + (1 - qmdp_state.done_flags) * gamma * next_values

    def critic_loss(params):
        _values = \
            train_state.critic_state.apply_fn(
                params,
                qmdp_state.states,
                qmdp_state.actions,
                qmdp_state.time_idxs
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
    qmdp_state: QMDPState,
    alpha: float
) -> tuple[JointTrainState, Array]:
    def actor_loss(params):
        _, actions, log_probs, _ = \
            train_state.policy_state.apply_fn(
                rng_key=rng_key,
                params=params,
                carry=qmdp_state.carry,
                observation=qmdp_state.observations
            )
        values = \
            train_state.critic_state.apply_fn(
                train_state.critic_state.params,
                qmdp_state.states,
                actions,
                qmdp_state.time_idxs
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


def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    qmdp_state: QMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)
    train_state, critic_loss = critic_train_step(critic_key, train_state, qmdp_state, alpha, gamma)
    train_state, policy_loss = policy_train_step(policy_key, train_state, qmdp_state, alpha)
    train_state = critic_target_update(train_state, tau=0.005)
    return train_state, policy_loss, critic_loss


@partial(
    jax.jit,
    static_argnames=("env_obj", "buffer_obj", "policy_network", "num_steps", "num_particles", "alpha", "gamma"),
    donate_argnames=("buffer_state", "pomdp_state", "train_state"),
)
def step_and_train(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    pomdp_state: POMDPState,
    buffer_obj: UniformSamplingQueue,
    buffer_state: ReplayBufferState,
    train_state: JointTrainState,
    policy_network: PolicyNetwork,
    num_steps: int,
    num_particles: int,
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
            policy_network=policy_network,
            num_particles=num_particles,
        )
        _qmdp_state = get_qmdp_state(_pomdp_state)
        _buffer_state = buffer_obj.insert(_buffer_state, _qmdp_state)
        _buffer_state, _qmdp_state_sample = buffer_obj.sample(_buffer_state)

        _train_state, _, _ = \
            gradient_step(
                rng_key=_train_key,
                train_state=_train_state,
                qmdp_state=_qmdp_state_sample,
                alpha=alpha,
                gamma=gamma
            )
        return (_pomdp_state, _buffer_state, _train_state), None

    keys = random.split(rng_key, num_steps)
    (pomdp_state, buffer_state, train_state), _ = jax.lax.scan(
        body, (pomdp_state, buffer_state, train_state), keys
    )
    return pomdp_state, buffer_state, train_state
