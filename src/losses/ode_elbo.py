"""
ELBO loss for the DMM model, both continuous and discrete observations
"""
from typing import Dict, Union, Tuple, List

import torch
from torch import nn

from .utils import gaussian_kl, gaussian_nll


class ODEContinuousELBO(nn.Module):
    """
    Compute the ELBO loss for the Latent ODE model with continuous observations

    Parameters
    ----------
    annealing_params : Dict[str, Union[bool, float]]
        Structured as follows:
        {
            "enabled": True, # whether to use annealing
            "warm_up": 0, # number of epochs to wait before starting annealing
            "n_epochs_for_full": 100, # number of epochs to wait
                                       # (after warm_up) before reaching full KL weight
        }
    rmse_eval_latent : bool, by default False
        Whether to evaluate RMSE on the latent space
        (not just the observation space)
    """

    def __init__(
        self,
        annealing_params: Dict[str, Union[bool, float]],
        z0_mean: float = 0,
        z0_log_var: float = 0,
    ):
        super().__init__()
        self.annealing_params = annealing_params
        self.z0_mean = torch.tensor(z0_mean, requires_grad=False)
        self.z0_log_var = torch.tensor(z0_log_var, requires_grad=False)

    def forward(
        self,
        model: nn.Module,
        data: Union[torch.Tensor, List[torch.Tensor]],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for the loss

        Parameters
        ----------
        data : torch.Tensor
            Ground truth observations, x_{1:T}
            shape = (batch_size, time_steps, obs_dim)
        epoch : int, by default 0
            Which epoch this loss is in, needed for annealing

        Returns
        -------
        torch.Tensor
            float value for the loss
        Dict[str, float]
            Dictionary of losses for each component of the ELBO
            Used for logging
        """
        observation_gt = data[0]

        # model forward function
        z0_mu, z0_log_var = model.inference_model(observation_gt)
        z0 = z0_mu + torch.randn_like(z0_mu) * torch.exp(0.5 * z0_log_var)
        transition_samples = model.transition_model(z0, 24, 0.1)
        emission_distribution = model.emission_model(transition_samples)

        logp_obs_loss = (
            gaussian_nll(
                emission_distribution[0],
                emission_distribution[1],
                observation_gt,
            ).mean()
            * observation_gt.shape[2]
        )

        kl_loss = (
            gaussian_kl(z0_mu, z0_log_var, self.z0_mean, self.z0_log_var).mean()
            * z0_mu.shape[-1]
        )

        annealing_factor = 1
        if self.annealing_params["enabled"]:
            if epoch < self.annealing_params["warm_up"]:
                annealing_factor = 0
            else:
                annealing_factor = min(
                    (epoch - self.annealing_params["warm_up"])
                    / self.annealing_params["n_epochs_for_full"],
                    1,
                )
        total_loss = logp_obs_loss + annealing_factor * kl_loss
        logging = {
            "Training loss": total_loss.item(),
            "log_p observation loss": logp_obs_loss.item(),
            "KL loss": kl_loss.item(),
            "logp + kl loss": logp_obs_loss.item() + kl_loss.item(),
            "annealing_factor": annealing_factor,
            "epoch": epoch,
        }
        return total_loss, logging

    def rolling_window_eval(
        self,
        model: nn.Module,
        data: torch.Tensor,
        eval_window_shifts: List[int],
        n_eval_windows: int,
    ) -> Dict[int, List[float]]:
        """
        Rolling window evaluation of RMSE

        Parameters
        ----------
        model : nn.Module
            Model to use for prediction
        data : torch.Tensor
            Data to use for evaluation, observations, shape = (batch_size, time_steps, obs_dim)
        eval_window_shifts : List[int]
            List of shifts to use for evaluation
        n_eval_windows : int
            Number of times to evaluate (each time shifts by 1)

        Returns
        -------
        Dict[int, List[float]]
            Dictionary of RMSE for each shift
        """
        max_window_shift = max(eval_window_shifts)
        rolling_window_rmse = {i: [] for i in eval_window_shifts}

        all_obs = data[0]

        for n in range(n_eval_windows):
            n_observations = all_obs.shape[1] - n - max_window_shift
            if n_observations <= 0:
                print(
                    "Warning: during RMSE rolling window evaluation, n_observations <= 0"
                )
                continue
            observations = all_obs[:, :n_observations, :]
            preds, _ = model.predict_future(observations, max_window_shift)
            for shift in eval_window_shifts:
                shift_preds = preds[:, n_observations + shift - 1, :]
                shift_obs = all_obs[:, n_observations + shift - 1, :]
                rmse = torch.sqrt(torch.mean((shift_preds - shift_obs) ** 2)).item()
                rolling_window_rmse[shift].append(rmse)

        return rolling_window_rmse
