from typing import NamedTuple


class SAC(NamedTuple):
    total_time_steps: int = 100000
    buffer_size: int = 100000
    learning_starts: int = 5000
    policy_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005
