"""
Base class for state space models
"""
from typing import Callable, Tuple, Any

import torch
from torch import nn


class StateSpaceModel(nn.Module):
    """
    Base class for state space models

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

    def __init__(
        self,
        inference_model: Callable[[torch.Tensor], Tuple[Any, torch.Tensor]],
        emission_model: Callable[[torch.Tensor], Any],
        transition_model: Callable[[torch.Tensor], Any],
    ):
        super().__init__()
        self.emission_model = emission_model
        self.transition_model = transition_model
        self.inference_model = inference_model

    def forward(self, observations: torch.Tensor):
        """
        Forward pass through the model, mostly used for training

        Parameters
        ----------
        observations : torch.Tensor
            Observations, shape = (batch_size, n_time_steps, obs_dim)

        Returns
        -------
        latent_distribution: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Output from the inference model, output shape depends on the model
        emission_distribution: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Output from the emission model, output shape depends on the model
        transition_distribution: torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Output from the transition model, output shape depends on the model
        """
        latent_distribution, latent_samples = self.inference_model(observations)
        emission_distribution = self.emission_model(latent_samples)
        transition_distribution = self.transition_model(latent_samples)

        return latent_distribution, emission_distribution, transition_distribution

    def predict_future(
        self, observations: torch.Tensor, n_future_time_steps: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future observations and latent variables

        Parameters
        ----------
        observations : torch.Tensor
            Observations,
            shape = (batch_size, n_time_steps, obs_dim)
        n_future_time_steps : int
            Number of future time steps to predict

        Returns
        -------
        observation_samples : torch.Tensor
            Observations samples,
            shape = (batch_size, n_time_steps + n_future_time_steps, obs_dim)
        latent_samples : torch.Tensor
            Latent variable samples,
            shape = (batch_size, n_time_steps + n_future_time_steps, latent_dim)
        """
        _, latent_samples = self.inference_model(
            observations
        )  # latent_samples shape = (batch_size, n_time_steps, latent_dim)

        # Future predictions using the transition model
        all_latent_samples = latent_samples
        if n_future_time_steps > 0:
            future_latents = [latent_samples[:, -1:, :]]
            for _ in range(n_future_time_steps):
                transition_distribution = self.transition_model(future_latents[-1])
                future_latents.append(
                    self.transition_model.sample(transition_distribution)
                )
            future_latents = torch.cat(future_latents[1:], dim=1)

            all_latent_samples = torch.cat([latent_samples, future_latents], dim=1)

        emission_distribution = self.emission_model(all_latent_samples)
        predictions = self.emission_model.sample(emission_distribution)

        return predictions, all_latent_samples
