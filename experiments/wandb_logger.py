from typing import Dict, Any
import os

import wandb
from flax.struct import dataclass


class WandbLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: dataclass,
        log_dir: str,
    ):
        """Initialize the Weights & Biases logger.
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Name of this specific experiment run
            config: Dictionary of hyperparameters and configuration or a NamedTuple
            log_dir: Directory to save local logs
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Convert config to dictionary if it's a NamedTuple
        config_dict = config.__dict__
            
        self.wandb_run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
            dir=self.log_dir,  # Set wandb to use the specified log directory
        )

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        wandb.log(metrics, step=step)

    def finish(self):
        """Finish the wandb run."""
        wandb.finish()
