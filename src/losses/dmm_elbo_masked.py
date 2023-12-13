"""
ELBO loss for the DMM model, both continuous and discrete observations
"""
from typing import Dict, Union, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from .utils import gaussian_kl, gaussian_nll, impute_transition_distribution
import numpy as np

class DMMContinuousELBOMasked(nn.Module):
    """
    Compute the ELBO loss for the DMM model, as described in section 3 in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al., 2016

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
        rmse_eval_latent: bool = False,
        z0_mean: float = 0,
        z0_log_var: float = 0,
    ):
        super().__init__()
        self.annealing_params = annealing_params
        self.rmse_eval_latent = rmse_eval_latent
        self.z0_mean = z0_mean
        self.z0_log_var = z0_log_var

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
        mask = data[1]

        # model forward function
        latent_distribution, latent_samples = model.inference_model(observation_gt)
        emission_distribution = model.emission_model(latent_samples)
        latent_samples2 = latent_distribution[0] + torch.randn_like(
            latent_distribution[1]
        ) * torch.exp(0.5 * latent_distribution[1])
        transition_distribution = model.transition_model(latent_samples2)

        # Add prior for t=0 and drop last prediction
        transition_distribution = impute_transition_distribution(
            transition_distribution, self.z0_mean, self.z0_log_var
        )

        logp_obs_loss = (
            gaussian_nll(
                emission_distribution[0],
                emission_distribution[1],
                observation_gt,
            )*mask
            * observation_gt.shape[2]
        )
        logp_obs_loss = logp_obs_loss.mean()

        kl_mask = mask.sum(dim=2) > 0
        kl_mask = kl_mask.float().reshape(-1, mask.shape[1], 1)
        kl_loss = ( 
            gaussian_kl(*latent_distribution, *transition_distribution)*kl_mask
            * latent_distribution[0].shape[2]
        )
        kl_loss = kl_loss.mean()

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
        if self.rmse_eval_latent and len(eval_window_shifts) > 1:
            raise ValueError(
                "Can only evaluate RMSE on latent space if eval_window_shifts = [0]"
            )

        all_obs = data[0]
        mask = data[1]

        for n in range(n_eval_windows):
            n_observations = all_obs.shape[1] - n - max_window_shift
            if n_observations <= 0:
                print(
                    "Warning: during RMSE rolling window evaluation, n_observations <= 0"
                )
                continue
            observations = all_obs[:, :n_observations, :]
            if self.rmse_eval_latent:
                inf_out, _ = model.inference_model(all_obs)
                latents = inf_out[0]
                shift_preds = latents
                # shift_gt = all_latent
                # rmse = torch.sqrt(torch.mean((shift_preds - shift_gt) ** 2)).item()
                # rolling_window_rmse[0].append(rmse)
            else:
                preds, _ = model.predict_future(observations, max_window_shift)
                for shift in eval_window_shifts:
                    shift_preds = preds[:, n_observations + shift - 1, :]
                    shift_obs = all_obs[:, n_observations + shift - 1, :]
                    diff_sq = (shift_preds - shift_obs) ** 2
                    diff_sq *= mask[:, n_observations + shift - 1, :].float()
                    rmse = diff_sq.float().sum()
                    rmse /= mask[:, n_observations + shift - 1, :].float().sum()
                    rolling_window_rmse[shift].append(torch.sqrt(rmse).item())

        return rolling_window_rmse
