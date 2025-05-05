from typing import Dict, Any, Optional, List, NamedTuple

import wandb


class WandbLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        experiment_group: Optional[str] = None,
        experiment_tags: Optional[List[str]] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
        logger_directory: str = "logs"
    ):
        """Initialize the Weights & Biases logger.

        Args:
            project_name: Name of the Weights & Biases project
            experiment_name: Name of the experiment
            experiment_group: Optional group name for organizing experiments
            experiment_tags: Optional list of tags for the experiment
            experiment_config: Optional dictionary of configuration parameters
            logger_directory: Directory to store logs
        """
        # Initialize wandb run with optional group and tags
        init_kwargs = {
            "project": project_name,
            "name": experiment_name,
            "dir": logger_directory,
            "settings": wandb.Settings(start_method="thread"),
        }

        if experiment_group:
            init_kwargs["group"] = experiment_group

        if experiment_tags:
            init_kwargs["tags"] = experiment_tags

        if experiment_config:
            init_kwargs["config"] = experiment_config

        self.wandb_run = wandb.init(**init_kwargs)

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names and values
        """
        wandb.log(metrics)

    def finish(self):
        """Finish the wandb run."""
        wandb.finish()
