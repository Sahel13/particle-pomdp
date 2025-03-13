from typing import Callable

from jax import Array, numpy as jnp
from flax import linen as nn

from distrax import (
    Chain,
    MultivariateNormalDiag,
    Transformed,
    Block
)

from baselines.sac.base import PRNGKey, Parameters


class MLP(nn.Module):
    layer_sizes: tuple[int, ...]

    @nn.compact
    def __call__(self, x: Array):

        for size in self.layer_sizes[:-1]:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.layer_sizes[-1])(x)


class CriticNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    layer_sizes: tuple[int, ...] = (256, 256)
    num_critics: int = 2

    @nn.compact
    def __call__(self, state: Array, action: Array, time: Array):
        feat = self.feature_fn(state)
        time = time / self.time_norm
        x = jnp.concatenate([feat, action, time[..., None]], -1)
        values = [MLP(self.layer_sizes + (1,))(x) for _ in range(self.num_critics)]
        return jnp.concatenate(values, axis=-1)


class PolicyNetwork(nn.Module):
    feature_fn: Callable
    time_norm: int
    layer_sizes: tuple[int, ...] = (256, 256, 1)
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, state: Array, time: Array) -> [Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.layer_sizes[-1])

        feat = self.feature_fn(state)
        time = time / self.time_norm
        x = jnp.concatenate([feat, time[..., None]], -1)
        for size in self.layer_sizes[:-1]:
            x = nn.relu(nn.Dense(size)(x))
        return nn.Dense(self.layer_sizes[-1])(x), log_std


def policy_sample(
    rng_key: PRNGKey,
    state: Array,
    time: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain
) -> tuple[Array, Array]:
    mean, log_std = network.apply({"params": params}, state, time)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    return dist.sample(seed=rng_key)


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    state: Array,
    time: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain
) -> tuple[Array, Array, Array]:
    mean, log_std = network.apply({"params": params}, state, time)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return action, log_prob, bijector.forward(mean)


# def create_gauss_policy(
#     network: PolicyNetwork,
#     bijector: Chain
# ) -> GaussianPolicy:
#
#     def sample(
#         rng_key: PRNGKey,
#         state: Array,
#         time: Array,
#         params: Parameters,
#     ) -> tuple[Array, Array]:
#         mean, log_std = network.apply({"params": params}, state, time)
#         base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
#         dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.dim))
#         return dist.sample(seed=rng_key)
#
#     def sample_and_log_prob(
#         rng_key: PRNGKey,
#         state: Array,
#         time: Array,
#         params: Parameters,
#     ) -> tuple[Array, Array]:
#         mean, log_std = network.apply({"params": params}, state, time)
#         base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
#         dist = Transformed(distribution=base, bijector=Block(bijector, ndims=network.dim))
#         action, log_prob = dist.sample_and_log_prob(seed=rng_key)
#         return action, log_prob
#
#     def init(
#         rng_key: PRNGKey,
#         state_dim: int,
#         batch_dim: int,
#         learning_rate: float,
#     ) -> TrainState:
#         state_key, param_key = random.split(rng_key, 2)
#         dummy_state = random.normal(state_key, (batch_dim, state_dim))
#         dummy_time = jnp.zeros((batch_dim,))
#         init_params = network.init(param_key, dummy_state, dummy_time)["params"]
#         train_state = TrainState.create(
#             apply_fn=network.apply,
#             params=init_params,
#             tx=optax.adam(learning_rate)
#         )
#         return train_state
#
#     return GaussianPolicy(
#         dim=network.dim,
#         sample=sample,
#         sample_and_log_prob=sample_and_log_prob,
#         init=init
#     )
