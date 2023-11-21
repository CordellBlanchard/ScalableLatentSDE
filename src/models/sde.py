from typing import Callable, Tuple, Any, List

import torch
from torch import nn

from .base import StateSpaceModel
from .modules import (
    EmissionNormalBase,
    EmissionNetworkNormal,
    StructuredInferenceLR,
    SDETransitionTimeIndep,
)


class SDEBase(StateSpaceModel):
    """
    Base class for State Space Models with transition models as SDEs

    Parameters
    ----------
    inference_model : Callable[[torch.Tensor], Tuple[Any, torch.Tensor]]
        Inference model takes in observations and returns the latent distribution and latent samples
    emission_model : Callable[[torch.Tensor], Any]
        Emission model takes in latent samples and returns
        the emission distribution (or samples from it)
    transition_model : Callable[[torch.Tensor], Any]
        Transition model takes in latent samples and returns
        the transition distribution (or samples from it)
    """

    def predict_future(
        self, inputs: Tuple[Any, torch.Tensor], time_steps: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future observations and latent variables

        Parameters
        ----------
        inputs : Tuple[Any, torch.Tensor]
            inputs[0] is observations, inputs[1] is time_steps
            Observations shape = (batch_size, n_time_steps, obs_dim)
        time_steps: torch.Tensor
            Time steps of the current latent variables, and future time values to predict
            shape=(batch_size, n_time_steps + n_future_time_steps)

        Returns
        -------
        observation_samples : torch.Tensor
            Observations samples,
            shape = (batch_size, n_time_steps + n_future_time_steps, obs_dim)
        latent_samples : torch.Tensor
            Latent variable samples,
            shape = (batch_size, n_time_steps + n_future_time_steps, latent_dim)
        """
        assert len(inputs) == 2
        observations, _ = inputs

        _, latent_samples = self.inference_model(
            observations
        )  # latent_samples shape = (batch_size, n_time_steps, latent_dim)
        batch_size, _, latent_dim = latent_samples.shape

        # Future predictions using the transition model
        all_latent_samples = latent_samples
        if time_steps is not None and time_steps.shape[1] > latent_samples.shape[1]:
            future_latents = [latent_samples[:, -1:, :]]
            for t in range(time_steps.shape[1] - latent_samples.shape[1]):
                start_times = time_steps[:, latent_samples.shape[1] + t - 1].reshape(
                    -1, 1
                )
                end_times = time_steps[:, latent_samples.shape[1] + t].reshape(-1, 1)
                dt = (end_times - start_times) / self.transition_model.n_euler_steps
                transition_distribution = self.transition_model(
                    future_latents[-1][:, 0], start_times, dt
                )
                future_latents.append(
                    self.transition_model.sample(transition_distribution).reshape(
                        batch_size, 1, latent_dim
                    )
                )
            future_latents = torch.cat(future_latents[1:], dim=1)
            all_latent_samples = torch.cat([latent_samples, future_latents], dim=1)

        emission_distribution = self.emission_model(all_latent_samples)
        predictions = self.emission_model.sample(emission_distribution)

        return predictions, all_latent_samples


class SDEContinuousFixedEmission(SDEBase):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Both emission and transition models are fixed to the ground truth distributions
    Note: emission sigma is set to 1 instead of 20 as in the paper, found this to work better
    """

    def __init__(self, n_euler_steps: int, transition_hidden_size: int):
        latent_dim = 1
        obs_dim = 1
        emission_mean = lambda latent: 0.5 * latent
        emission_sigma = lambda latent: torch.log(torch.ones_like(latent) * 20)
        emission_model = EmissionNormalBase(emission_mean, emission_sigma)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, hidden_dim=256, n_layers=2
        )

        transition_model = SDETransitionTimeIndep(
            latent_dim, hidden_size=transition_hidden_size, n_euler_steps=n_euler_steps
        )
        super().__init__(inference_model, emission_model, transition_model)


class SDEContinuous(SDEBase):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Both emission and transition models are fixed to the ground truth distributions
    Note: emission sigma is set to 1 instead of 20 as in the paper, found this to work better
    """

    def __init__(
        self,
        n_euler_steps: int,
        transition_hidden_size: int,
        latent_dim: int = 1,
        obs_dim: int = 1,
    ):
        emission_model = EmissionNetworkNormal(latent_dim, obs_dim, hidden_size=10)
        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, hidden_dim=256, n_layers=2
        )

        transition_model = SDETransitionTimeIndep(
            latent_dim, hidden_size=transition_hidden_size, n_euler_steps=n_euler_steps
        )
        super().__init__(inference_model, emission_model, transition_model)
