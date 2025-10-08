from typing import List, Optional, NamedTuple


class SLAC(NamedTuple):
    num_belief_particles: int = 32
    total_time_steps: int = int(1e6)
    buffer_size: int = int(1e6)
    learning_starts: int = 5000
    policy_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 16
    alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005


class SLACExperiment(NamedTuple):
    # Experiment settings
    env_id: str
    num_seeds: int = 10
    starting_seed: int = 0
    cuda_device: str = "0"

    # Algorithm settings
    num_belief_particles: int = 32
    total_time_steps: int = int(1e6)
    buffer_size: int = int(1e6)
    learning_starts: int = 5000
    policy_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 16
    alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005

    # Logger settings
    use_logger: bool = True
    project_name: str = "particle-pomdp"
    experiment_group: str = "slac"
    experiment_tags: Optional[List[str]] = ["slac"]
    experiment_id: Optional[str] = None
    logger_directory: str = "logs"
