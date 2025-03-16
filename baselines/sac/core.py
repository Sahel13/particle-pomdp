from typing import Dict, NamedTuple

from flax.training.train_state import TrainState


class JointTrainState(NamedTuple):
    policy_state: TrainState
    critic_state: TrainState
    critic_target_params: Dict


class SACConfig(NamedTuple):
    seed: int = 1
    total_timesteps: int = int(1e5)
    buffer_size: int = int(1e5)
    batch_size: int = 256
    learning_starts: int = int(5e3)
    policy_lr: float = 3e-4
    critic_lr: float = 1e-3
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005
