"""
Similar to ELBO loss for the SDE model, both continuous and discrete observations
"""
from typing import Dict, Union, Tuple, List

import torch
from torch import nn
import torch.distributions as td


class SDEContinuousELBO(nn.Module):
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
    clipping_params : Dict[str, Union[bool, float]]
        Structured as follows:
        {
            "enabled": True, # whether to use clipping
            "clip_max": 100, # maximum value for the clipped loss
        }
    prior_factor : float
        Weight of the prior MSE loss
    """

    def __init__(
        self,
        annealing_params: Dict[str, Union[bool, float]],
        clipping_params: Dict[str, Union[bool, float]],
        prior_factor: float,
    ):
        super().__init__()
        self.annealing_params = annealing_params
        self.clipping_params = clipping_params
        self.prior_factor = prior_factor

    def forward(
        self,
        latent_distribution: Tuple[torch.Tensor, torch.Tensor],
        emission_distribution: Tuple[torch.Tensor, torch.Tensor],
        transitions_distribution: Tuple[torch.Tensor, torch.Tensor],
        inputs: torch.Tensor,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for the loss

        Parameters
        ----------
        latent_distribution : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the latent distribution, q(z_t | z_{t-1}, x_{t:T})
            shape = (batch_size, time_steps, latent_dim)
        emission_distribution : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the emission distribution, p(x_t | z_t)
            shape = (batch_size, time_steps, obs_dim)
        transitions_distribution : List[torch.Tensor]
            First element: samples from the prior distribution, p(z_t | z_{t-1})
            Second element: mean at multiple time steps between each t-1 and t
            Third element: standard deviation at multiple time steps between each t-1 and t
        inputs : torch.Tensor
            inputs[0]: Ground truth observations, x_{1:T}
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
        observation_gt = inputs[0]
        logp_obs_loss = 0
        observation_distribution = td.normal.Normal(
            emission_distribution[0], emission_distribution[1]
        )  # shape = (batch_size, n_time_steps, obs_dim)
        logp_obs_loss = -1 * observation_distribution.log_prob(observation_gt).mean()

        batch_size, n_time_steps, latent_dim = latent_distribution[0].shape

        posterior_mu = latent_distribution[0][:, 1:]
        posterior_sigma = latent_distribution[1][:, 1:]
        posterior_distribution = td.normal.Normal(posterior_mu, posterior_sigma)
        z_samples = transitions_distribution[0][-1].reshape(
            batch_size, (n_time_steps - 1), latent_dim
        )
        kl_loss = -1 * posterior_distribution.log_prob(z_samples).mean()

        drift = torch.cat(transitions_distribution[1], dim=0)
        diffusion = torch.cat(transitions_distribution[2], dim=0)

        drift_prior = torch.cat(transitions_distribution[0][1:], dim=0)
        prior_mse = torch.mean((drift + drift_prior) ** 2 / diffusion)

        if self.clipping_params["enabled"]:
            kl_loss_clipped = torch.clamp(
                kl_loss,
                max=self.clipping_params["clip_max"],
                min=0,
            )
        else:
            kl_loss_clipped = kl_loss

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
        total_loss = (
            logp_obs_loss
            + annealing_factor * kl_loss_clipped
            + self.prior_factor * prior_mse
        )
        logging = {
            "Training loss": total_loss.item(),
            "log_p observation loss": logp_obs_loss.item(),
            "KL loss": kl_loss.item(),
            "KL loss clipped": kl_loss_clipped.item(),
            "logp + kl loss": logp_obs_loss.item() + kl_loss.item(),
            "annealing_factor": annealing_factor,
            "prior_mse": prior_mse.item(),
        }
        return total_loss, logging

    def rolling_window_eval(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        eval_window_shifts: List[int],
        n_eval_windows: int,
    ) -> Dict[int, List[float]]:
        """
        Rolling window evaluation of RMSE

        Parameters
        ----------
        model : nn.Module
            Model to use for prediction
        data : Tuple[torch.Tensor, torch.Tensor]
            Inputs, first element is observations, shape = (batch_size, time_steps, obs_dim)
            Second element is time steps, shape = (batch_size, time_steps)
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
        all_obs, all_times = data
        for n in range(n_eval_windows):
            n_observations = all_obs.shape[1] - n - max_window_shift
            if n_observations <= 0:
                print(
                    "Warning: during RMSE rolling window evaluation, n_observations <= 0"
                )
                continue
            observations = all_obs[:, :n_observations, :]
            cur_time_steps = all_times[:, :n_observations]
            fut_time_steps = all_times[:, : n_observations + max_window_shift]
            preds, _ = model.predict_future(
                [observations, cur_time_steps], fut_time_steps
            )
            for shift in eval_window_shifts:
                shift_preds = preds[:, n_observations + shift - 1, :]
                shift_obs = all_obs[:, n_observations + shift - 1, :]
                rmse = torch.sqrt(torch.mean((shift_preds - shift_obs) ** 2)).item()
                rolling_window_rmse[shift].append(rmse)
        return rolling_window_rmse
