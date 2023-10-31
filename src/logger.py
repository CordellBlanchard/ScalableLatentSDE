"""
Logger classes for training and evaluation
"""
from typing import Any, Dict
import wandb
import yaml


class PrintLogger:
    """
    Simple logger that prints to console

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        print("Training Configuration:")
        print(yaml.dump(config, allow_unicode=True, default_flow_style=False))

    def log_train(self, to_log: Dict[str, Any], epoch: int) -> None:
        """
        Print nothing during training
        """
        return

    def log_eval(self, to_log: Dict[str, Any], epoch: int) -> None:
        """
        Print evaluation metrics during training

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        epoch : int
            Current epoch
        """
        print(f"Validation Metrics, Epoch: {epoch}")
        print(yaml.dump(to_log, allow_unicode=True, default_flow_style=False))

    def log_test(self, to_log: Dict[str, Any]) -> None:
        """
        Print test metrics after training is complete

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        """
        print("Training Completed")
        print("Test Metrics:")
        print(yaml.dump(to_log, allow_unicode=True, default_flow_style=False))


class WandbLogger:
    """
    Weights and Biases logger

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        self.run = wandb.init(
            project="state_space_models", entity="ml-healthcare-project", config=config
        )

    def log_train(self, to_log: Dict[str, Any], epoch: int):
        """
        Log training metrics

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        epoch : int
            Current epoch
        """
        self.run.log(
            {"train": to_log},
        )

    def log_eval(self, to_log: Dict[str, Any], epoch: int):
        """
        Log validation metrics during training

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        epoch : int
            Current epoch
        """
        self.log_metrics(to_log, "val")

    def log_test(self, to_log: Dict[str, Any]) -> None:
        """
        Log test metrics after training is complete

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        """
        self.log_metrics(to_log, "test")
        self.run.finish()

    def log_metrics(self, to_log: Dict[str, Any], prefix: str) -> None:
        """
        Log evaluation metrics

        Parameters
        ----------
        to_log : Dict[str, Any]
            Dictionary of metrics to log
        prefix : str
            Prefix for the metric name
        """
        rolling_window_rmse = to_log["rolling_window"]

        data = []
        for key, value in rolling_window_rmse.items():
            data.append([int(key), value])

        table = wandb.Table(data=data, columns=["Window Shift", "RMSE"])
        wandb.log(
            {
                f"{prefix}_rolling_window": wandb.plot.line(
                    table,
                    "Window Shift",
                    "RMSE",
                    title=f"{prefix} Rolling Window RMSE with different shifts",
                )
            }
        )
        for key, value in to_log.items():
            if key != "rolling_window":
                wandb.log({f"{prefix}_{key}": value})
