import wandb
from typing import Dict, Any, Optional
from flax.struct import dataclass


class WandbLogger:
    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: dataclass,
        log_dir: Optional[str] = None,
    ):
        """Initialize the Weights & Biases logger.
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Name of this specific experiment run
            config: Dictionary of hyperparameters and configuration or a NamedTuple
            log_dir: Optional directory to save local logs
        """
        self.log_dir = log_dir
        
        # Convert config to dictionary if it's a NamedTuple
        config_dict = config.__dict__
            
        self.wandb_run = wandb.init(
            project=project_name,
            name=experiment_name,
            config=config_dict,
            settings=wandb.Settings(start_method="thread"),
        )
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        wandb.log(metrics, step=step)
        
    def log_histogram(self, name: str, values: Any, step: Optional[int] = None):
        """Log a histogram to wandb.
        
        Args:
            name: Name of the histogram
            values: Values to create histogram from
            step: Optional step number
        """
        wandb.log({name: wandb.Histogram(values)}, step=step)
        
    def log_image(self, name: str, image: Any, step: Optional[int] = None):
        """Log an image to wandb.
        
        Args:
            name: Name of the image
            image: Image data (numpy array or PIL Image)
            step: Optional step number
        """
        wandb.log({name: wandb.Image(image)}, step=step)
        
    def log_plot(self, name: str, figure: Any, step: Optional[int] = None):
        """Log a matplotlib figure to wandb.
        
        Args:
            name: Name of the plot
            figure: Matplotlib figure object
            step: Optional step number
        """
        wandb.log({name: wandb.Image(figure)}, step=step)
        
    def finish(self):
        """Finish the wandb run."""
        wandb.finish()
