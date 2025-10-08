from typing import List, Optional, NamedTuple


class P3O(NamedTuple):
    # Algorithm hyperparameters
    total_time_steps: int = 1000000
    num_history_particles: int = 128
    num_belief_particles: int = 32
    slew_rate_penalty: float = 5e-2
    tempering: float = 0.3
    backward_sampling: bool = True
    backward_sampling_mult: float = 2

    # damped version
    damping: Optional[float] = 0.0

    # learning hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 16
    init_std: float = 1.0


class P3OExperiment(NamedTuple):
    # Environment settings
    env_id: str
    num_seeds: int = 10
    starting_seed: int = 0
    cuda_device: str = "0"

    # Algorithm hyperparameters
    total_time_steps: int = int(1e6)
    num_history_particles: int = 128
    num_belief_particles: int = 32
    slew_rate_penalty: float = 5e-2
    tempering: float = 0.3
    backward_sampling: bool = True
    backward_sampling_mult: float = 2

    # damped version
    damping: Optional[float] = 0.0

    # learning hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 16
    init_std: float = 1.0

    # Logger settings
    use_logger: bool = True
    project_name: str = "particle-pomdp"
    experiment_group: str = "p3o"
    experiment_tags: Optional[List[str]] = ["p3o", "test"]
    experiment_id: Optional[str] = None
    logger_directory: str = "logs"
