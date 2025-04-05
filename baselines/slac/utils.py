from distrax import Chain, MultivariateNormalDiag, Transformed
from jax import Array
from jax import numpy as jnp
from flax import struct

from baselines.slac.arch import PolicyNetwork
from ppomdp.core import Carry, Parameters, PRNGKey


@struct.dataclass
class SLACConfig:
    # Environment settings
    seed: int = 0
    env_id: str = "cartpole"

    # Algorithm hyperparameters
    num_belief_particles: int = 32
    total_time_steps: int = 25000
    buffer_size: int = 100000
    learning_starts: int = 5000
    policy_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005
    
    # Logger settings
    use_logger: bool = True
    project_name: str = "particle-pomdp"
    experiment_name: str = "slac-cartpole-seed-0"
    log_dir: str = "logs"


def policy_sample_and_log_prob(
    rng_key: PRNGKey,
    carry: list[Carry],
    observation: Array,
    params: Parameters,
    network: PolicyNetwork,
    bijector: Chain,
) -> tuple[list[Carry], Array, Array, Array]:
    carry, mean, log_std = network.apply({"params": params}, carry, observation)
    base = MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    dist = Transformed(distribution=base, bijector=bijector)
    action, log_prob = dist.sample_and_log_prob(seed=rng_key)
    return carry, action, log_prob, bijector.forward(mean)