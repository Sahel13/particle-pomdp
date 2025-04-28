from typing import List, Optional, NamedTuple


class DVRL(NamedTuple):
    num_belief_particles: int = 32
    total_time_steps: int = 100000
    buffer_size: int = 100000
    learning_starts: int = 5000
    policy_lr: float = 0.0001
    critic_lr: float = 0.001
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005


class DVRLExperiment(NamedTuple):
    # Experiment settings
    env_id: str = "cartpole"
    num_seeds: int = 10
    cuda_device: str = "0"

    # Algorithm hyperparameters
    num_belief_particles: int = 32
    total_time_steps: int = 100000
    buffer_size: int = 100000
    learning_starts: int = 5000
    policy_lr: float = 0.0001
    critic_lr: float = 0.001
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005

    # Logger settings
    use_logger: bool = True
    project_name: str = "particle-pomdp"
    experiment_group: Optional[str] = "dvrl-cartpole"
    experiment_tags: Optional[List[str]] = ["dsmc", "cartpole", "test"]
    logger_directory: str = "logs"
