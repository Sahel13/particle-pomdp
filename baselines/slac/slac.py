from copy import deepcopy
from functools import partial

import jax
import optax
from distrax import Block
from flax.training.train_state import TrainState
from jax import Array, random, numpy as jnp

from ppomdp.core import BeliefState, Carry, PRNGKey
from ppomdp.envs.core import POMDPEnv, POMDPState
from ppomdp.bijector import Tanh
from ppomdp.policy.arch import GRUEncoder, DualHeadMLPDecoder
from ppomdp.utils import custom_split
from ppomdp.smc.utils import initialize_belief, update_belief

from baselines.common import (
    JointTrainState,
    sample_hidden_states,
    sample_random_actions,
)
from baselines.slac.arch import CriticNetwork, PolicyNetwork
from baselines.slac.utils import policy_sample_and_log_prob


def _pomdp_base(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    states: Array,
    carry: list[Carry],
    observations: Array,
    belief_states: BeliefState,
    random_actions: bool,
) -> tuple[Array, list[Carry], Array, BeliefState, Array]:

    # Sample action.
    key, action_key = random.split(rng_key, 2)
    if random_actions:
        next_carry = carry
        actions = sample_random_actions(action_key, env_obj)
    else:
        next_carry, actions, _, _ = policy_state.apply_fn(
            rng_key=action_key,
            carry=carry,
            observation=observations,
            params=policy_state.params,
        )

    # Sample next state.
    key, state_keys = custom_split(key, env_obj.num_envs + 1)
    next_states = jax.vmap(env_obj.trans_model.sample)(state_keys, states, actions)

    # Sample observation.
    key, obs_keys = custom_split(key, env_obj.num_envs + 1)
    next_observations = jax.vmap(env_obj.obs_model.sample)(obs_keys, next_states)

    # Update belief.
    belief_keys = random.split(key, env_obj.num_envs)
    next_belief_states = jax.vmap(update_belief, (0, None, None, 0, 0, 0))(
        belief_keys,
        env_obj.trans_model,
        env_obj.obs_model,
        belief_states,
        next_observations,
        actions,
    )

    return next_states, next_carry, next_observations, next_belief_states, actions


def pomdp_init(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    num_belief_particles: int,
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

    carry = policy_network.reset(env_obj.num_envs)

    key, step_key = random.split(key, 2)
    next_states, next_carry, next_observations, next_belief_states, actions = (
        _pomdp_base(
            rng_key=step_key,
            env_obj=env_obj,
            policy_state=policy_state,
            states=states,
            carry=carry,
            observations=observations,
            belief_states=belief_states,
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


def pomdp_step(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    pomdp_state: POMDPState,
    num_belief_particles: int,
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
                policy_state=policy_state,
                states=states,
                carry=carry,
                observations=observations,
                belief_states=belief_states,
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
            rng_key=rng_key,
            env_obj=env_obj,
            policy_state=policy_state,
            policy_network=policy_network,
            num_belief_particles=num_belief_particles,
            random_actions=random_actions,
        )

    return jax.lax.cond(
        jnp.all(pomdp_state.done_flags == 0), _true_fn, _false_fn, pomdp_state
    )


def critic_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    policy_network: PolicyNetwork,
    pomdp_states: POMDPState,
    alpha: float,
    gamma: float,
) -> tuple[JointTrainState, Array]:

    def critic_loss(params):
        def body(val, args):
            loss, key, carry = val
            actions, belief_states, next_belief_states, \
                next_observations, rewards, done_flags, time_idxs, = args

            key, action_key, state_key, next_state_key = random.split(key, 4)
            next_carry, next_actions, next_log_probs, _ = (
                train_state.policy_state.apply_fn(
                    rng_key=action_key,
                    carry=carry,
                    observation=next_observations,
                    params=train_state.policy_state.params,
                )
            )
            states = sample_hidden_states(
                rng_key=state_key,
                particles=belief_states.particles,
                weights=belief_states.weights,
            )
            next_states = sample_hidden_states(
                rng_key=next_state_key,
                particles=next_belief_states.particles,
                weights=next_belief_states.weights,
            )

            next_values = train_state.critic_state.apply_fn(
                train_state.critic_target_params,
                next_states,
                next_actions,
                time_idxs + 1,
            )
            next_values = jnp.min(next_values, axis=-1) - alpha * next_log_probs

            target_values = jax.lax.stop_gradient(
                rewards + (1 - done_flags) * gamma * next_values
            )

            values = train_state.critic_state.apply_fn(
                params,
                states,
                actions,
                time_idxs,
            )
            error = values - jnp.expand_dims(target_values, -1)
            loss += 0.5 * jnp.mean(jnp.square(error))
            return (loss, key, next_carry), None

        _, batch_size, _ = pomdp_states.observations.shape
        init_carry = policy_network.reset(batch_size)
        init_observations = pomdp_states.observations[0, ...]

        # Pass the first observation to the policy. This is because the policy
        # will be used to predict the action corresponding to `next_observation`
        # inside the loss.
        key, sub_key = random.split(rng_key)
        carry, *_ = train_state.policy_state.apply_fn(
            sub_key,
            init_carry,
            init_observations,
            train_state.policy_state.params,
        )

        init_loss = jnp.array(0.0)
        (loss, _, _), _ = jax.lax.scan(
            f=body,
            init=(init_loss, key, carry),
            xs=(
                pomdp_states.actions,
                pomdp_states.belief_states,
                pomdp_states.next_belief_states,
                pomdp_states.next_observations,
                pomdp_states.rewards,
                pomdp_states.done_flags,
                pomdp_states.time_idxs,
            )
        )
        return loss

    grad_fn = jax.value_and_grad(critic_loss)
    loss, grads = grad_fn(train_state.critic_state.params)
    critic_state = train_state.critic_state.apply_gradients(grads=grads)
    train_state = train_state._replace(critic_state=critic_state)
    return train_state, loss


def policy_train_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    policy_network: PolicyNetwork,
    pomdp_states: POMDPState,
    alpha: float,
) -> tuple[JointTrainState, Array]:

    def actor_loss(params):
        def body(val, args):
            loss, key, carry = val
            observations, belief_states, time_idxs = args

            key, action_key, state_key = random.split(key, 3)
            next_carry, actions, log_probs, _ = \
                train_state.policy_state.apply_fn(
                    action_key,
                    carry,
                    observations,
                    params
                )
            states = sample_hidden_states(
                rng_key=state_key,
                particles=belief_states.particles,
                weights=belief_states.weights,
            )
            values = train_state.critic_state.apply_fn(
                train_state.critic_state.params,
                states,
                actions,
                time_idxs
            )
            min_values = jnp.min(values, axis=-1)
            loss += jnp.mean(alpha * log_probs - min_values)
            return (loss, key, next_carry), None

        _, batch_size, _ = pomdp_states.observations.shape
        init_carry = policy_network.reset(batch_size)
        init_loss = jnp.array(0.0)

        (loss, _, _), _ = jax.lax.scan(
            f=body,
            init=(init_loss, rng_key, init_carry),
            xs=(
                pomdp_states.observations,
                pomdp_states.belief_states,
                pomdp_states.time_idxs
            )
        )
        return loss

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


@partial(jax.jit, static_argnames=("policy_network", "alpha", "gamma", "tau"))
def gradient_step(
    rng_key: PRNGKey,
    train_state: JointTrainState,
    policy_network: PolicyNetwork,
    pomdp_states: POMDPState,
    alpha: float,
    gamma: float,
    tau: float,
) -> tuple[JointTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key)

    # Have (num_time_steps, batch_size) as the first two dimensions
    pomdp_states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), pomdp_states)

    # Update critic
    train_state, critic_loss = critic_train_step(
        critic_key, train_state, policy_network, pomdp_states, alpha, gamma
    )

    # Update policy
    train_state, policy_loss = policy_train_step(
        policy_key, train_state, policy_network, pomdp_states, alpha
    )

    # Update target critic
    train_state = critic_target_update(train_state, tau)
    return train_state, policy_loss, critic_loss


def create_train_state(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_lr: float,
    critic_lr: float,
) -> tuple[JointTrainState, PolicyNetwork, CriticNetwork]:

    policy_encoder = GRUEncoder(
        feature_fn=lambda x: x,
        dense_sizes=(256, 256),
        recurr_sizes=(128, 128),
        use_layer_norm=True,
    )
    policy_decoder = DualHeadMLPDecoder(
        decoder_sizes=(256, 256),
        output_dim=env_obj.action_dim,
    )
    policy_network = PolicyNetwork(
        encoder=policy_encoder,
        decoder=policy_decoder,
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
        tx=optax.adam(critic_lr),
    )
    joint_state = JointTrainState(
        policy_train_state,
        critic_train_state,
        critic_target_params,
    )
    return joint_state, policy_network, critic_networks


@partial(
    jax.jit,
    static_argnames=(
        "env_obj",
        "policy_network",
        "num_belief_particles",
        "random_actions",
    ),
)
def pomdp_rollout(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
    policy_state: TrainState,
    policy_network: PolicyNetwork,
    num_belief_particles: int,
    random_actions: bool,
):
    """Simulate trajectories with the SLAC policy."""
    def body(pomdp_state, key):
        pomdp_state = pomdp_step(
            rng_key=key,
            env_obj=env_obj,
            policy_state=policy_state,
            policy_network=policy_network,
            pomdp_state=pomdp_state,
            num_belief_particles=num_belief_particles,
            random_actions=random_actions,
        )
        return pomdp_state, pomdp_state

    init_key, scan_keys = custom_split(rng_key, env_obj.num_time_steps + 1)
    init_pomdp_state = pomdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        policy_state=policy_state,
        policy_network=policy_network,
        num_belief_particles=num_belief_particles,
        random_actions=random_actions,
    )
    _, pomdp_states = jax.lax.scan(body, init_pomdp_state, scan_keys)

    pomdp_states = jax.tree.map(
        lambda x, y: jnp.concatenate((x[None], y), axis=0), init_pomdp_state, pomdp_states
    )
    # Swap time and num_envs axes
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), pomdp_states)
