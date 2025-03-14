from argparse import ArgumentParser
from functools import partial

import jax
import optax

from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from distrax import Block
from ppomdp.bijector import Tanh

from baselines.sac.base import PRNGKey, SACEnv, SACEnvState, SACTrainState, SACConfig
from baselines.sac.arch import PolicyNetwork, CriticNetwork, policy_sample_and_log_prob
from baselines.sac.envs import PendulumEnv, CartPoleEnv, LightDark2DEnv

from brax.training.replay_buffers import UniformSamplingQueue

from copy import deepcopy
import matplotlib.pyplot as plt

# jax.config.update("jax_disable_jit", True)


def sample_random_actions(
    rng_key: PRNGKey,
    env: SACEnv,
) -> Array:
    return random.uniform(
        key=rng_key,
        shape=(env.num_envs, env.action_dim),
        minval=-1.0,
        maxval=1.0
    )


def init_env(
    rng_key: PRNGKey,
    env: SACEnv,
    policy_state: TrainState,
    random_actions: bool = False,
) -> SACEnvState:

    key, state_key, action_key = random.split(rng_key, 3)
    state = env.prior_dist.sample(seed=state_key, sample_shape=(env.num_envs,))
    time = jnp.zeros(env.num_envs)

    action = sample_random_actions(action_key, env) if random_actions \
        else policy_state.apply_fn(action_key, params=policy_state.params, state=state, time=time)[0]

    reward = jax.vmap(env.reward_fn)(state, action, time)

    keys = random.split(key, env.num_envs)
    next_state = jax.vmap(env.trans_model.sample)(keys, state, action)
    done = jnp.zeros(env.num_envs)

    return SACEnvState(
        state=state,
        action=action,
        next_state=next_state,
        reward=reward,
        total_reward=reward.copy(),
        time=time,
        done=done,
    )


@partial(jax.jit, static_argnums=(1, 4), donate_argnums=3)
def step_env(
    rng_key: PRNGKey,
    env: SACEnv,
    env_state: SACEnvState,
    policy_state: TrainState,
    random_actions: bool = False,
) -> SACEnvState:
    num_samples = env.num_envs

    def _true_fn(_env_state):
        key, action_key = random.split(rng_key, 2)
        state = _env_state.next_state
        time = (_env_state.time + 1)

        action = sample_random_actions(action_key, env) if random_actions \
            else policy_state.apply_fn(action_key, params=policy_state.params, state=state, time=time)[0]

        state_keys = random.split(key, num_samples)
        reward = jax.vmap(env.reward_fn)(state, action, time)
        total_reward = _env_state.total_reward + reward
        next_state = jax.vmap(env.trans_model.sample)(state_keys, state, action)
        done = jnp.where(time == env.num_time_steps, 1., 0.)

        return SACEnvState(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            total_reward=total_reward,
            time=time,
            done=done
        )

    def _false_fn(_outer_state):
        return init_env(rng_key, env, policy_state, random_actions)

    return jax.lax.cond(
        jnp.all(env_state.done == 0.), _true_fn, _false_fn, env_state
    )


def create_train_state(
    rng_key: PRNGKey,
    env: SACEnv,
    policy_lr: float,
    critic_lr: float,
) -> SACTrainState:
    policy_network = PolicyNetwork(
        feature_fn=env.feature_fn,
        time_norm=env.num_time_steps,
        layer_sizes=(256, 256, env.action_dim),
        init_log_std=nn.initializers.constant(1.0),
    )
    critic_networks = CriticNetwork(
        feature_fn=env.feature_fn,
        time_norm=env.num_time_steps,
        layer_sizes=(256, 256),
        num_critics=2,
    )

    dummy_states = jnp.empty((1, env.state_dim))
    dummy_actions = jnp.empty((1, env.action_dim))
    dummy_time = jnp.empty((1,))

    critic_key, policy_key = random.split(rng_key)
    policy_params = policy_network.init(policy_key, dummy_states, dummy_time)["params"]
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
    return SACTrainState(
        policy_train_state,
        critic_train_state,
        critic_target_params
    )


def critic_train_step(
    rng_key: PRNGKey,
    train_state: SACTrainState,
    env_state: SACEnvState,
    alpha: float,
    gamma: float,
) -> tuple[SACTrainState, Array]:
    next_action, next_log_prob, _ = \
        train_state.policy_state.apply_fn(
            rng_key=rng_key,
            params=train_state.policy_state.params,
            state=env_state.next_state,
            time=env_state.time,
        )
    _next_values = \
        train_state.critic_state.apply_fn(
            train_state.critic_target_params,
            env_state.next_state,
            next_action,
            env_state.time + 1,
        )
    next_value = jnp.min(_next_values, axis=-1) - alpha * next_log_prob
    target_value = env_state.reward + (1 - env_state.done) * gamma * next_value

    def critic_loss(params):
        _value = train_state.critic_state.apply_fn(params, env_state.state, env_state.action, env_state.time)
        _error = _value - jnp.expand_dims(target_value, -1)
        return 0.5 * jnp.mean(jnp.square(_error))

    grad_fn = jax.value_and_grad(critic_loss)
    loss, grads = grad_fn(train_state.critic_state.params)
    _critic_state = train_state.critic_state.apply_gradients(grads=grads)
    train_state = train_state._replace(critic_state=_critic_state)
    return train_state, loss


def policy_train_step(
    rng_key: PRNGKey,
    train_state: SACTrainState,
    env_state: SACEnvState,
    alpha: float
) -> tuple[SACTrainState, Array]:
    def policy_loss(params):
        action, log_prob, _ = \
            train_state.policy_state.apply_fn(
                rng_key=rng_key,
                params=params,
                state=env_state.state,
                time=env_state.time
            )
        values = \
            train_state.critic_state.apply_fn(
                train_state.critic_state.params,
                env_state.state,
                action,
                env_state.time,
            )
        min_value = jnp.min(values, axis=-1)
        return jnp.mean(alpha * log_prob - min_value)

    grad_fn = jax.value_and_grad(policy_loss)
    loss, grads = grad_fn(train_state.policy_state.params)
    _policy_state = train_state.policy_state.apply_gradients(grads=grads)
    train_state = train_state._replace(policy_state=_policy_state)
    return train_state, loss


def critic_target_update(
    train_state: SACTrainState,
    tau: float
) -> SACTrainState:
    updated_params = jax.tree.map(
        lambda param, target: tau * param + (1 - tau) * target,
        train_state.critic_state.params,
        train_state.critic_target_params,
    )
    return train_state._replace(critic_target_params=updated_params)


def gradient_step(
    rng_key: PRNGKey,
    train_state: SACTrainState,
    env_state: SACEnvState,
    alpha: float,
    gamma: float,
) -> tuple[SACTrainState, Array, Array]:
    critic_key, policy_key = random.split(rng_key, 2)
    train_state, critic_loss = critic_train_step(critic_key, train_state, env_state, alpha, gamma)
    train_state, policy_loss = policy_train_step(policy_key, train_state, env_state, alpha)
    train_state = critic_target_update(train_state, tau=0.005)
    return train_state, critic_loss, policy_loss


if __name__ == "__main__":
    config = SACConfig()

    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name",
        choices=["pendulum", "cartpole", "lightdark2d"],
        default="lightdark2d",
    )
    args = parser.parse_args()

    if args.env == "pendulum":
        env = PendulumEnv
    elif args.env == "cartpole":
        env = CartPoleEnv
    else:
        env = LightDark2DEnv

    key = random.key(config.seed)
    key, sub_key = random.split(key)
    train_state = create_train_state(sub_key, env, config.policy_lr, config.critic_lr)

    key, sub_key = random.split(key)
    env_state = init_env(sub_key, env, train_state.policy_state, True)

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], env_state)
    buffer = UniformSamplingQueue(
        config.buffer_size,
        buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    @partial(jax.jit, static_argnames="num_steps")
    def step_and_train(
        rng_key,
        init_train_state,
        init_env_state,
        init_buffer_state,
        num_steps
    ):

        def body(carry, rng_keys):
            train_state, env_state, buffer_state = carry
            step_key, train_key = rng_keys

            env_state = step_env(step_key, env, env_state, train_state.policy_state)
            buffer_state = buffer.insert(buffer_state, env_state)

            buffer_state, _env_state_sample = buffer.sample(buffer_state)
            train_state, _, _ = \
                gradient_step(
                    rng_key=train_key,
                    train_state=train_state,
                    env_state=_env_state_sample,
                    alpha=config.alpha,
                    gamma=config.gamma
                )

            return (train_state, env_state, buffer_state), None

        return jax.lax.scan(
            body,
            (init_train_state, init_env_state, init_buffer_state),
            random.split(rng_key, num_steps * 2).reshape((num_steps, 2)),
        )[0]

    key, sub_key = random.split(key)
    buffer_state = buffer.init(sub_key)
    buffer.insert_internal = jax.jit(buffer.insert_internal)
    buffer_state = buffer.insert(buffer_state, env_state)

    # Pre-populate the buffer with random trajectories.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        env_state = step_env(sub_key, env, env_state, train_state.policy_state, True)
        buffer_state = buffer.insert(buffer_state, env_state)
        if jnp.all(env_state.done == 1):
            print(
                f"Step: {global_step:7d} | "
                + f"Episodic reward: {env_state.total_reward.mean():10.2f}"
            )

    # Ensure that training starts with a fresh episode.
    env_state = env_state._replace(done=jnp.ones(env.num_envs))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env.num_time_steps + 1)

    # Training loop.
    for global_step in range(
        config.learning_starts, config.total_timesteps, steps_per_epoch
    ):
        key, sub_key = random.split(key)
        train_state, env_state, buffer_state = \
            step_and_train(sub_key, train_state, env_state, buffer_state, steps_per_epoch)

        print(
            f"Step: {global_step + steps_per_epoch:7d} | "
            + f"Episodic reward: {env_state.total_reward.mean():10.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Evaluate the learned policy.
    key, state_key = random.split(key)
    state = env.prior_dist.sample(seed=state_key)

    def body_fn(carry, rng_key):
        action_key, state_key = random.split(rng_key)
        state, time = carry
        _, _, action = train_state.policy_state.apply_fn(
            rng_key=action_key,
            params=train_state.policy_state.params,
            state=state,
            time=time
        )
        state = env.trans_model.sample(state_key, state, action)
        return (state, time + 1), (state, action)

    _, (states, actions) = jax.lax.scan(
        body_fn, (state, 0), random.split(key, env.num_time_steps)
    )
    states = jnp.concatenate([state[None, ...], states], axis=0)

    if args.env == "pendulum":
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
    elif args.env == "cartpole":
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle("Simulated trajectory")

        axs[0].plot(states[:, 1])
        axs[0].set_ylabel("Angle")
        axs[0].grid(True)

        axs[1].plot(states[:, 3])
        axs[1].set_ylabel("Angular velocity")
        axs[1].grid(True)

        axs[2].plot(actions[:, 0])
        axs[2].set_ylabel("Action")
        axs[2].set_xlabel("Time")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    elif args.env == "lightdark2d":
        plt.figure()
        plt.title("Simulated trajectory")
        plt.plot(states[:, 0], states[:, 1], "g-")
        plt.plot(2, 2, "ro", label="Starting location")
        plt.plot(0, 0, "rx", label="Target location")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.axis("equal")
        plt.show()

        plt.figure()
        plt.plot(actions[:, 0])
        plt.plot(actions[:, 1])
        plt.xlabel("Time")
        plt.ylabel("Action")
        plt.show()
