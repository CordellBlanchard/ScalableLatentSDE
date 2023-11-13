"""
ELBO loss for the DMM model, both continuous and discrete observations
"""
from typing import Dict, Union, Tuple, List

import torch
from torch import nn
import torch.distributions as td


class DMMContinuousELBO(nn.Module):
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
    """

    def __init__(self, annealing_params: Dict[str, Union[bool, float]]):
        super().__init__()
        self.annealing_params = annealing_params

    def forward(
        self,
        latent_distribution: Tuple[torch.Tensor, torch.Tensor],
        emission_distribution: Tuple[torch.Tensor, torch.Tensor],
        transition_distribution: Tuple[torch.Tensor, torch.Tensor],
        observation_gt: torch.Tensor,
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
        transition_distribution : Tuple[torch.Tensor, torch.Tensor]
            Mean and standard deviation of the transition distribution, p(z_t | z_{t-1})
            shape = (batch_size, time_steps, latent_dim)
        observation_gt : torch.Tensor
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
        # Add prior for t=0 and drop last prediction
        transition_distribution = (
            torch.cat(
                [
                    torch.zeros_like(transition_distribution[0][:, 0:1, :]),
                    transition_distribution[0][:, :-1],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    torch.ones_like(transition_distribution[1][:, 0:1, :]),
                    transition_distribution[1][:, :-1],
                ],
                dim=1,
            ),
        )

        logp_obs_loss = 0
        kl_loss = 0
        n_time_steps = latent_distribution[0].shape[1]  # number of time steps
        for t in range(n_time_steps):
            observation_distribution = td.normal.Normal(
                emission_distribution[0][:, t], emission_distribution[1][:, t]
            )  # shape = (batch_size, obs_dim)
            posterior_distribution = td.normal.Normal(
                latent_distribution[0][:, t], latent_distribution[1][:, t]
            )  # shape = (batch_size, latent_dim)
            prior_distribution = td.normal.Normal(
                transition_distribution[0][:, t], transition_distribution[1][:, t]
            )  # shape = (batch_size, latent_dim)

            logp_obs_loss += observation_distribution.log_prob(
                observation_gt[:, t]
            ).mean()

            kl_loss -= td.kl_divergence(
                posterior_distribution, prior_distribution
            ).mean()

        logp_obs_loss = -1 * logp_obs_loss / n_time_steps
        kl_loss = -1 * kl_loss / n_time_steps

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
        for n in range(n_eval_windows):
            n_observations = data.shape[1] - n - max_window_shift
            if n_observations <= 0:
                print(
                    "Warning: during RMSE rolling window evaluation, n_observations <= 0"
                )
                continue
            observations = data[:, :n_observations, :]
            preds, _ = model.predict_future(observations, max_window_shift)
            for shift in eval_window_shifts:
                shift_preds = preds[:, n_observations + shift - 1, :]
                shift_obs = data[:, n_observations + shift - 1, :]
                rmse = torch.sqrt(torch.mean((shift_preds - shift_obs) ** 2)).item()
                rolling_window_rmse[shift].append(rmse)
        return rolling_window_rmse
