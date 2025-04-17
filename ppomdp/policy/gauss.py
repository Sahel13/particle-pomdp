from functools import partial
from typing import Callable, Union, List

import jax
import optax

from jax import Array, random, numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from distrax import (
    Bijector,
    MultivariateNormalDiag,
    Uniform,
    Transformed,
)

from ppomdp.core import (
    Carry,
    RecurrentPolicy,
    Parameters,
    HistoryParticles,
    PRNGKey
)
from ppomdp.arch import (
    LSTMEncoder,
    GRUEncoder,
    MLPDecoder
)


class RecurrentNeuralGaussPolicy(nn.Module):
    """
    Neural policy module for processing sequences with recurrent encoding and dense decoding layers.

    Attributes:
        encoder (Union[ppomdp.arch.LSTMEncoder, ppomdp.arch.GRUEncoder]): Recurrent encoder module.
        decoder (Decoder): Dense decoder module.
        init_log_std (Callable): Initializer for the log standard deviation parameter.
    """

    encoder: Union[LSTMEncoder, GRUEncoder]
    decoder: MLPDecoder
    init_log_std: Callable = nn.initializers.ones

    @nn.compact
    def __call__(self, carry: list[Carry], s: Array) -> tuple[list[Carry], Array, Array]:
        log_std = self.param("log_std", self.init_log_std, self.decoder.output_dim)

        next_carry, s = self.encoder(carry, s)
        s = self.decoder(s)
        return next_carry, s, log_std

    @property
    def dim(self):
        return self.decoder.output_dim

    def reset(self, batch_size: int) -> list[Carry]:
        return self.encoder.reset(batch_size)


def create_recurrent_neural_gauss_policy(
    network: RecurrentNeuralGaussPolicy,
    bijector: Bijector
) -> RecurrentPolicy:
    """
    Creates a squashed neural policy that conforms to the RecurrentPolicy interface.

    Args:
        network (RecurrentNeuralGaussPolicy): The neural network used for the policy
        bijector (Chain): policy bijector to enforce action limits

    Returns:
        RecurrentPolicy: A policy that implements the RecurrentPolicy interface
    """

    def reset(batch_size: int) -> list[Carry]:
        return network.reset(batch_size)

    def sample(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, mean, log_std = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return next_carry, dist.sample(seed=rng_key)

    def sample_and_log_prob(
        rng_key: PRNGKey,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array, Array]:
        next_carry, mean, log_std = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        action, log_prob = dist.sample_and_log_prob(seed=rng_key)
        return next_carry, action, log_prob

    def carry_and_log_prob(
        action: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> tuple[list[Carry], Array]:
        next_carry, mean, log_std = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return next_carry, dist.log_prob(action)

    def log_prob(
        actions: Array,
        carry: list[Carry],
        observations: Array,
        params: Parameters,
    ) -> Array:
        _, mean, log_std = network.apply({"params": params}, carry, observations)
        base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = Transformed(distribution=base, bijector=bijector)
        return dist.log_prob(actions)

    @jax.jit
    def pathwise_carry(
        init_carry: list[Carry],
        observations: Array,
        params: Parameters,
    ):
        def concat_trees(x, y):
            return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

        def body(carry, obs):
            next_carry, _, _ = network.apply({"params": params}, carry, obs)
            return next_carry, next_carry

        _, all_carry = jax.lax.scan(body, init_carry, observations)
        return concat_trees(init_carry, all_carry)

    def pathwise_log_prob(
        particles: HistoryParticles,
        params: Parameters
    ) -> Array:
        def body(t, log_probs):
            actions = particles.actions[t]
            observations = particles.observations[t - 1]
            carry = jax.tree.map(lambda x: x[t - 1], particles.carry)
            log_prob_incs = log_prob(actions, carry, observations, params)
            return log_probs + log_prob_incs

        num_time_steps, batch_size, _ = particles.actions.shape
        init_log_probs = jnp.zeros(batch_size)
        log_probs = jax.lax.fori_loop(1, num_time_steps, body, init_log_probs)
        return log_probs

    def entropy(params: Parameters) -> Array:
        sigma = jnp.diag(jnp.exp(2. * params["log_std"]))
        return 0.5 * (
            network.dim * jnp.log(2.0 * jnp.pi * jnp.exp(1))
            + jnp.linalg.slogdet(sigma)[1]
        )

    def init(
        rng_key: PRNGKey,
        state_dim: int,
        action_dim: int,
        batch_dim: int,
        learning_rate: float
    ) -> TrainState:
        input_key, param_key = random.split(rng_key, 2)
        dummy_carry = network.reset(batch_dim)
        dummy_input = random.normal(input_key, (batch_dim, state_dim))
        init_params = network.init(param_key, dummy_carry, dummy_input)["params"]
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=init_params,
            tx=optax.adam(learning_rate)
        )
        return train_state

    return RecurrentPolicy(
        dim=network.dim,
        reset=reset,
        sample=sample,
        log_prob=log_prob,
        pathwise_carry=pathwise_carry,
        pathwise_log_prob=pathwise_log_prob,
        sample_and_log_prob=sample_and_log_prob,
        carry_and_log_prob=carry_and_log_prob,
        entropy=entropy,
        init=init
    )


@partial(jax.jit, static_argnames="policy")
def train_recurrent_neural_gauss_policy_pathwise(
    policy: RecurrentPolicy,
    train_state: TrainState,
    particles: HistoryParticles,
    damping: float = 1.0
) -> tuple[TrainState, Array]:
    def loss_fn(params):
        log_probs = damping * policy.pathwise_log_prob(particles, params)
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss


@partial(jax.jit, static_argnames="policy")
def train_recurrent_neural_gauss_policy_stepwise(
    policy: RecurrentPolicy,
    train_state: TrainState,
    actions: Array,
    carry: List[Carry],
    observations: Array,
    damping: float = 1.0
):
    def loss_fn(params):
        log_probs = damping * policy.log_prob(
            actions=actions,
            carry=carry,
            observations=observations,
            params=params
        )
        return -1.0 * jnp.mean(log_probs)

    loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss
