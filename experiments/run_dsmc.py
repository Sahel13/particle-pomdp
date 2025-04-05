import jax
import tyro

from brax.training.replay_buffers import UniformSamplingQueue
from jax import numpy as jnp
from jax import random

from baselines.dsmc import (
    DSMCConfig,
    create_train_state,
    pomdp_init,
    pomdp_step,
    step_and_train,
)
from wandb_logger import WandbLogger
from common import get_env


if __name__ == "__main__":
    config = tyro.cli(DSMCConfig)

    # Set up wandb logging if enabled
    logger = None
    if config.use_logger:
        logger = WandbLogger(
            project_name=config.project_name,
            experiment_name=config.experiment_name,
            config=config,
            log_dir=config.log_dir
        )

    env_obj = get_env(config.env_id)

    key = random.key(config.seed)
    key, sub_key = random.split(key)
    train_state, _, _ = create_train_state(sub_key, env_obj, config)

    key, init_key = random.split(key)
    pomdp_state = pomdp_init(
        rng_key=init_key,
        env_obj=env_obj,
        alg_cfg=config,
        train_state=train_state,
        random_actions=True,
    )

    # Set up the replay buffer from Brax.
    buffer_entry_prototype = jax.tree.map(lambda x: x[0], pomdp_state)
    buffer_obj = UniformSamplingQueue(
        config.buffer_size,
        buffer_entry_prototype,
        sample_batch_size=config.batch_size
    )

    buffer_obj.insert_internal = jax.jit(buffer_obj.insert_internal)
    buffer_obj.sample_internal = jax.jit(buffer_obj.sample_internal)

    key, buffer_key = random.split(key)
    buffer_state = buffer_obj.init(buffer_key)
    buffer_state = buffer_obj.insert(buffer_state, pomdp_state)

    # Pre-fill the buffer with random actions.
    for global_step in range(1, config.learning_starts):
        key, sub_key = random.split(key)
        pomdp_state = pomdp_step(
            rng_key=sub_key,
            env_obj=env_obj,
            alg_cfg=config,
            pomdp_state=pomdp_state,
            train_state=train_state,
            random_actions=True,
        )
        buffer_state = buffer_obj.insert(buffer_state, pomdp_state)

        if jnp.all(pomdp_state.done_flags == 1):
            expected_reward = pomdp_state.total_rewards.mean()

            if logger:
                logger.log_metrics({
                    "expected_reward": expected_reward,
                    "policy_log_std": train_state.policy_state.params['log_std'][0]
                }, step=global_step)
            print(
                f"Step: {global_step:6d} | Expected reward: {expected_reward:6.2f}"
            )

    # Ensure that training starts with a fresh episode.
    pomdp_state = pomdp_state._replace(done_flags=jnp.ones(env_obj.num_envs, dtype=jnp.int32))

    # Number of steps to take using the `lax.scan` loop (and how often to print training info).
    steps_per_epoch = 10 * (env_obj.num_time_steps + 1)

    # Training loop - slightly faster training with `jax.lax.scan`.
    for global_step in range(
        config.learning_starts, config.total_time_steps, steps_per_epoch
    ):
        key, sub_key = random.split(key)
        pomdp_state, buffer_state, train_state = step_and_train(
            sub_key,
            env_obj,
            config,
            pomdp_state,
            buffer_obj,
            buffer_state,
            train_state,
            steps_per_epoch,
        )
        expected_reward = pomdp_state.total_rewards.mean()
        if logger:
            logger.log_metrics({
                "expected_reward": expected_reward,
                "policy_log_std": train_state.policy_state.params['log_std'][0]
            }, step=global_step + steps_per_epoch)
        print(
            f"Step: {global_step + steps_per_epoch:6d} | "
            + f"Expected reward: {expected_reward:6.2f} | "
            + f"Policy log std: {train_state.policy_state.params['log_std'][0]:6.2f}"
        )

    # Finish wandb logging if enabled
    if logger:
        logger.finish()
