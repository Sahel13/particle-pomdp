import jax
from jax import Array, random, numpy as jnp
from distrax import Chain, MultivariateNormalDiag, Transformed

from ppomdp.core import PRNGKey, Parameters, Carry, InnerState
from ppomdp.envs.core import POMDPEnv, POMDPState, QMDPState
from baselines.slac.arch import PolicyNetwork


def sample_random_actions(
    rng_key: PRNGKey,
    env_obj: POMDPEnv,
) -> Array:
    return random.uniform(
        key=rng_key,
        shape=(env_obj.num_envs, env_obj.action_dim),
        minval=-1.0,
        maxval=1.0
    )


@jax.jit
def get_qmdp_state(pomdp_state: POMDPState) -> QMDPState:

    @jax.vmap
    def mean_belief(belief: InnerState) -> Array:
        return jnp.sum(belief.particles * belief.weights[..., None], axis=0)

    states = mean_belief(pomdp_state.beliefs)
    next_states = mean_belief(pomdp_state.next_beliefs)

    return QMDPState(
        states=states,
        carry=pomdp_state.carry,
        observations=pomdp_state.observations,
        actions=pomdp_state.actions,
        next_states=next_states,
        next_carry=pomdp_state.next_carry,
        next_observations=pomdp_state.next_observations,
        rewards=pomdp_state.rewards,
        time_idxs=pomdp_state.time_idxs,
        done_flags=pomdp_state.done_flags,
    )


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    carry: list[Carry],
    observation: Array,
    network: PolicyNetwork,
    params: Parameters,
    bijector: Chain,
) -> tuple[list[Carry], Array, Array, Array]:
    carry, mean, log_std = network.apply({"params": params}, carry, observation)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return carry, action, log_prob, bijector.forward(mean)
