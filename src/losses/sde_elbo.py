"""
Similar to ELBO loss for the SDE model, both continuous and discrete observations
"""
from typing import Dict, Union, Tuple, List

import torch
from torch import nn
import torch.distributions as td

from .utils import gaussian_nll, entropy_upper_bound


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
    entropy_weight : float
        Weight for the entropy term in the Pseudo-ELBO
    elog_weight : float
        Weight for the E[log q(z_t | z_{t-1})] term in the Pseudo-ELBO
    n_entropy_samples : int
        Number of samples to use for estimating the entropy of p(z_t | z_{t-1})
    """

    def __init__(
        self,
        annealing_params: Dict[str, Union[bool, float]],
        clipping_params: Dict[str, Union[bool, float]],
        entropy_weight: float,
        entropy_q_weight: float,
        elog_weight: float,
        n_entropy_samples: int,
        rmse_eval_latent: bool = False,
    ):
        super().__init__()
        self.annealing_params = annealing_params
        self.clipping_params = clipping_params
        self.entropy_weight = entropy_weight
        self.entropy_q_weight = entropy_q_weight
        self.elog_weight = elog_weight
        self.rmse_eval_latent = rmse_eval_latent
        self.n_entropy_samples = n_entropy_samples

    def forward(
        self,
        model: nn.Module,
        inputs: List[torch.Tensor],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass for the loss

        Parameters
        ----------
        model: nn.Module
            SDE model to use for prediction
        inputs : List[torch.Tensor]
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
        observations = inputs[0]
        time_steps = inputs[-1]

        latent_distribution, latent_samples = model.inference_model(observations)
        emission_distribution = model.emission_model(latent_samples)

        # Transition model forward pass
        batch_size, n_time_steps, latent_dim = latent_samples.shape
        start_times = time_steps[:, :-1].reshape(-1, 1)
        end_times = time_steps[:, 1:].reshape(-1, 1)
        dt = (end_times - start_times) / model.transition_model.n_euler_steps
        transition_distribution = model.transition_model(
            latent_samples[:, :-1].reshape(-1, latent_dim), start_times, dt
        )
        z_samples = transition_distribution[0][-1].reshape(
            batch_size, (n_time_steps - 1), latent_dim
        )

        logp_obs_loss = (
            gaussian_nll(
                emission_distribution[0],
                emission_distribution[1],
                observations,
            ).mean()
            * observations.shape[2]
        )

        E_p_log_q = (
            gaussian_nll(
                latent_distribution[0][:, 1:],
                latent_distribution[1][:, 1:],
                z_samples,
            ).mean()
            * latent_dim
        )

        entropy_q = (
            0.5
            * (latent_distribution[1] + torch.log(2 * torch.tensor(3.1415)) + 1).mean()
            * latent_distribution[0].shape[2]
        )

        # Estimate upper bound for entropy of p(z_t | z_{t-1})
        n_samples = self.n_entropy_samples
        to_pass = torch.ones((n_samples, batch_size, n_time_steps, latent_dim)).to(
            latent_samples.device
        )
        to_pass *= latent_samples.reshape(1, batch_size, n_time_steps, latent_dim)
        transition_distribution = model.transition_model(
            to_pass.reshape(batch_size * n_samples * n_time_steps, latent_dim),
            0,
            1e-1,
        )
        entropy_p_samples = transition_distribution[0][-1].reshape(
            n_samples, batch_size, n_time_steps, latent_dim
        )
        entropy_p = entropy_upper_bound(entropy_p_samples).mean()

        # kl_loss = -1*KL(p||q)
        kl_loss = self.elog_weight * E_p_log_q - self.entropy_weight * entropy_p

        if self.clipping_params["enabled"]:
            kl_loss_clipped = torch.clamp(
                kl_loss,
                max=self.clipping_params["clip_max"],
                min=self.clipping_params["clip_min"],
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
            + self.entropy_q_weight * entropy_q
        )
        logging = {
            "Training loss": total_loss.item(),
            "log_p observation loss": logp_obs_loss.item(),
            "KL loss": kl_loss.item(),
            "KL loss clipped": kl_loss_clipped.item(),
            "logp + kl loss": logp_obs_loss.item() + kl_loss.item(),
            "annealing_factor": annealing_factor,
            "entropy_q": entropy_q.item(),
            "entropy_p": entropy_p.item(),
            "E_p_log_q": E_p_log_q.item(),
            "epoch": epoch,
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
        all_obs = data[0]
        all_times = data[-1]
        if len(data) == 3:
            all_latents = data[1]

        for n in range(n_eval_windows):
            n_observations = all_obs.shape[1] - n - max_window_shift
            if n_observations <= 0:
                print(
                    "Warning: during RMSE rolling window evaluation, n_observations <= 0"
                )
                continue
            if self.rmse_eval_latent:
                inf_out, _ = model.inference_model(all_obs)
                latents = inf_out[0]
                shift_preds = latents
                shift_gt = all_latents
                rmse = torch.sqrt(torch.mean((shift_preds - shift_gt) ** 2)).item()
                rolling_window_rmse[0].append(rmse)
            else:
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
