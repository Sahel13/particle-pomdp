from typing import List, NamedTuple, Optional


class DSMC(NamedTuple):
    num_planner_steps: int = 10
    num_planner_particles: int = 32
    num_belief_particles: int = 32
    total_time_steps: int = 100000
    buffer_size: int = 100000
    learning_starts: int = 5000
    policy_lr: float = 0.0003
    critic_lr: float = 0.001
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 0.005


class DSMCExperiment(NamedTuple):
    # Experiment settings
    env_id: str
    num_seeds: int = 10
    cuda_device: str = "0"

    # Algorithm settings
    num_planner_steps: int = 10
    num_planner_particles: int = 32
    num_belief_particles: int = 32
    total_time_steps: int = int(1e6)
    buffer_size: int = int(1e6)
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    alpha: float = 0.2
    gamma: float = 0.95
    tau: float = 5e-3

    # Logger settings
    use_logger: bool = True
    project_name: str = "particle-pomdp"
    experiment_group: Optional[str] = "dsmc"
    experiment_tags: Optional[List[str]] = ["dsmc"]
    logger_directory: str = "logs"
