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

                data = data.to(device)

                model_output = self.model(data)
                loss, to_log = self.loss(*model_output, data, epoch)

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
        with torch.no_grad():
            eval_metrics = {}
            rolling_window_rmse = {i: [] for i in self.params["eval_window_shifts"]}
            for data in dataloader:
                data = data.to(self.params["device"])

                model_output = self.model(data)
                _, to_log = self.loss(*model_output, data, epoch)
                for key, value in to_log.items():
                    if key not in eval_metrics:
                        eval_metrics[key] = []

                    eval_metrics[key].append(value)

                # Moving window RMSE
                max_window_shift = max(self.params["eval_window_shifts"])
                n_eval_windows = self.params["n_eval_windows"]
                for n in range(n_eval_windows):
                    n_observations = data.shape[1] - n - max_window_shift
                    if n_observations <= 0:
                        print(
                            "Warning: during RMSE rolling window evaluation, n_observations <= 0"
                        )
                        continue
                    observations = data[:, :n_observations, :]
                    preds, _ = self.model.predict_future(observations, max_window_shift)
                    for shift in self.params["eval_window_shifts"]:
                        shift_preds = preds[:, n_observations + shift - 1, :]
                        shift_obs = data[:, n_observations + shift - 1, :]
                        rmse = torch.sqrt(
                            torch.mean((shift_preds - shift_obs) ** 2)
                        ).item()
                        rolling_window_rmse[shift].append(rmse)

            for key in rolling_window_rmse:
                rolling_window_rmse[key] = sum(rolling_window_rmse[key]) / len(
                    rolling_window_rmse[key]
                )
            for key in eval_metrics:
                eval_metrics[key] = sum(eval_metrics[key]) / len(eval_metrics[key])
            eval_metrics["rolling_window"] = rolling_window_rmse
            return eval_metrics
