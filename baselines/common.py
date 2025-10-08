from typing import NamedTuple, Dict, Union

import jax
from flax.training.train_state import TrainState
from jax import Array, random

from ppomdp.core import PRNGKey
from ppomdp.envs import pomdps
from ppomdp.envs.core import MDPEnv, POMDPEnv


def get_pomdp(env_name: str) -> POMDPEnv:
    if env_name == "pendulum":
        return pomdps.PendulumEnv
    elif env_name == "cartpole":
        return pomdps.CartPoleEnv
    elif env_name == "triangulation":
        return pomdps.TriangulationEnv
    elif env_name == "light-dark-2d":
        return pomdps.LightDark2DEnv
    else:
        raise NotImplementedError


class JointTrainState(NamedTuple):
    policy_state: TrainState
    critic_state: TrainState
    critic_target_params: Dict


def sample_random_actions(
    rng_key: PRNGKey,
    env_obj: Union[MDPEnv, POMDPEnv],
) -> Array:
    return random.uniform(
        key=rng_key,
        shape=(env_obj.num_envs, env_obj.action_dim),
        minval=-1.0,
        maxval=1.0
    )


def sample_hidden_states(
    rng_key: PRNGKey,
    particles: Array,
    weights: Array
) -> Array:
    batch_size, num_particles, _ = particles.shape

    def choice_fn(key, _particles, _weights):
        idx = random.choice(key, a=num_particles, p=_weights)
        return _particles[idx]

    keys = random.split(rng_key, batch_size)
    return jax.vmap(choice_fn)(keys, particles, weights)
