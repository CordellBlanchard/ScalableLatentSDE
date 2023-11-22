from typing import Callable, Tuple, Any, List

import torch
from torch import nn

from .base import StateSpaceModel
from .modules import (
    EmissionNetworkNormal,
    RNNInferenceNetwork,
    ODETransitionTimeIndep,
    ODEAdjoint,
)
from torchdiffeq import odeint_adjoint as odeint


class ODEBase(StateSpaceModel):
    """
    Base class for State Space Models with transition models as ODEs

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

        z0_mu, z0_log_var = self.inference_model(
            inputs
        )  # latent_samples shape = (batch_size, n_time_steps, latent_dim)
        z0 = z0_mu + torch.randn_like(z0_mu) * torch.exp(0.5 * z0_log_var)

        # Future predictions using the transition model
        all_latent_samples = self.transition_model(
            z0, inputs.shape[1] + time_steps - 1, 0.1
        )

        emission_distribution = self.emission_model(all_latent_samples)
        predictions = self.emission_model.sample(emission_distribution)

        return predictions, all_latent_samples


class ODEContinuous(ODEBase):
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        emission_hidden_dim: int = 64,
        inference_hidden_size: int = 64,
        n_inference_layers: int = 1,
        transition_hidden_dim: int = 64,
        n_euler_steps: int = 10,
    ):
        emission_model = EmissionNetworkNormal(latent_dim, obs_dim, emission_hidden_dim)

        rnn_inference_net = RNNInferenceNetwork(
            latent_dim,
            obs_dim,
            hidden_dim=inference_hidden_size,
            n_layers=n_inference_layers,
        )

        transition_model = ODETransitionTimeIndep(
            latent_dim, transition_hidden_dim, n_euler_steps
        )
        super().__init__(rnn_inference_net, emission_model, transition_model)


class ODEAdjointWrapper(nn.Module):
    def __init__(self, ode: nn.Module):
        super().__init__()
        self.ode = ode

    def forward(self, latents, max_time, dt):
        out = odeint(
            self.ode,
            latents,
            torch.arange(0, max_time + 1).float().to(latents.device),
            method="euler",
            options={"step_size": dt},
        ).permute(1, 0, 2)
        return out


class ODEContinuousAdjoint(ODEBase):
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        emission_hidden_dim: int = 64,
        inference_hidden_size: int = 64,
        n_inference_layers: int = 1,
        transition_hidden_dim: int = 64,
        n_euler_steps: int = 10,
    ):
        emission_model = EmissionNetworkNormal(latent_dim, obs_dim, emission_hidden_dim)

        rnn_inference_net = RNNInferenceNetwork(
            latent_dim,
            obs_dim,
            hidden_dim=inference_hidden_size,
            n_layers=n_inference_layers,
        )

        transition_model = ODEAdjointWrapper(
            ODEAdjoint(latent_dim, transition_hidden_dim, n_euler_steps)
        )
        super().__init__(rnn_inference_net, emission_model, transition_model)
