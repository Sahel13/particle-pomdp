from typing import List, Optional, NamedTuple


class NSMC(NamedTuple):
    # Algorithm hyperparameters
    total_time_steps: int = 1000000
    num_history_particles: int = 128
    num_belief_particles: int = 32
    slew_rate_penalty: float = 5e-2
    tempering: float = 0.3
    learning_rate: float = 3e-4
    batch_size: int = 256

    # Network architecture
    encoder_dense_sizes: tuple[int, ...] = (256, 256)
    encoder_recurr_sizes: tuple[int, ...] = (128, 128)
    decoder_dense_sizes: tuple[int, ...] = (256, 256)
    init_std: float = 1.0

    # damped version
    damping: Optional[float] = 0.0


class NSMCExperiment(NamedTuple):
    # Environment settings
    env_id: str = "cartpole"
    num_seeds: int = 10
    cuda_device: str = "0"

    # Algorithm hyperparameters
    total_time_steps: int = 1000000
    num_history_particles: int = 128
    num_belief_particles: int = 32
    slew_rate_penalty: float = 5e-2
    tempering: float = 0.3
    learning_rate: float = 3e-4
    batch_size: int = 256

    # Network architecture
    encoder_dense_sizes: tuple[int, ...] = (256, 256)
    encoder_recurr_sizes: tuple[int, ...] = (128, 128)
    decoder_dense_sizes: tuple[int, ...] = (256, 256)
    init_std: float = 1.0

    # damped version
    damping: Optional[float] = 0.0

    # Logger settings
    use_logger: bool = True
    project_name: str = "particle-pomdp"
    experiment_group: str = "nsmc-cartpole"
    experiment_tags: Optional[List[str]] = ["nsmc", "cartpole", "test"]
    logger_directory: str = "logs"
