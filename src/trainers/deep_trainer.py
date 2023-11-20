"""
Main trainer class for training all deep learning models
"""
from typing import Dict, Any
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


class DeepTrainer:
    """
    Trainer class for training models

    Parameters
    ----------
    model: nn.Module
        Model to train
    dataloaders: Dict[str, DataLoader]
        Dictionary of dataloaders for train, val, and test sets
    loss: nn.Module
        Loss function to use
        forward should take as input: model output, data, and epoch
    logger: Logger
        Used to log metrics during training
    trainer_params: Dict[str, Any]
        Dictionary of parameters for the trainer
        Should contain lr, n_epochs, val_freq, device, save_path, verbose
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        loss: nn.Module,
        logger: Any,
        trainer_params: Dict[str, Any],
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.loss = loss
        self.logger = logger
        self.params = trainer_params

    def run(self):
        """
        Run training and validation loop
        Save the model at the end
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        device = self.params["device"]

        self.model.to(device)

        self.model.train()
        for epoch in tqdm(
            range(self.params["n_epochs"]),
            disable=not self.params["verbose"],
            leave=False,
        ):
            for data in self.dataloaders["train"]:
                optimizer.zero_grad()

                if isinstance(data, list):
                    data = [d.to(device) for d in data]
                else:
                    data = data.to(device)

                loss, to_log = self.loss(self.model, data, epoch)

                loss.backward()
                optimizer.step()

                self.logger.log_train(to_log, epoch)

            if epoch % self.params["val_freq"] == 0:
                self.model.eval()
                to_log = self.run_eval(self.dataloaders["val"], epoch=epoch)
                self.logger.log_eval(to_log, epoch)
                self.model.train()

        # Eval on test set
        self.model.eval()
        to_log = self.run_eval(self.dataloaders["test"], epoch=self.params["n_epochs"])
        self.logger.log_test(to_log)

        # Save model
        torch.save(self.model.state_dict(), self.params["save_path"])

    def run_eval(self, dataloader, epoch: int):
        """
        Run evaluation loop

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Dataloader to use for evaluation
        epoch : int
            Current epoch, used in loss function

        Returns
        -------
        """
        device = self.params["device"]
        with torch.no_grad():
            eval_metrics = {}
            rolling_window_rmse = {i: [] for i in self.params["eval_window_shifts"]}
            for data in dataloader:
                if isinstance(data, list):
                    data = [d.to(device) for d in data]
                else:
                    data = data.to(device)

                _, to_log = self.loss(self.model, data, epoch)
                for key, value in to_log.items():
                    if key not in eval_metrics:
                        eval_metrics[key] = []

                    eval_metrics[key].append(value)

                # Moving window RMSE
                rolling_window_samples = self.loss.rolling_window_eval(
                    self.model,
                    data,
                    self.params["eval_window_shifts"],
                    self.params["n_eval_windows"],
                )

                for shift in self.params["eval_window_shifts"]:
                    rolling_window_rmse[shift].extend(rolling_window_samples[shift])

            for key in rolling_window_rmse:
                rolling_window_rmse[key] = sum(rolling_window_rmse[key]) / len(
                    rolling_window_rmse[key]
                )
            for key in eval_metrics:
                eval_metrics[key] = sum(eval_metrics[key]) / len(eval_metrics[key])
            eval_metrics["rolling_window"] = rolling_window_rmse
            return eval_metrics
