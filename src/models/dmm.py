"""
Implementations of the DMM for various experiements from the paper:
Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
"""
import torch
from torch import nn

from .base import StateSpaceModel
from .modules import (
    EmissionNormalBase,
    EmissionNetworkNormal,
    EmissionNetworkBinary,
    StructuredInferenceLR,
    TransformerSTLR,
    DeterministicTransitionFunction,
    GatedTransitionFunction,
)


class DMMContinuousFixedTheta(StateSpaceModel):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Both emission and transition models are fixed to the ground truth distributions

    Parameters
    ----------
    st_net_hidden_dim : int
        Dimension of the hidden layers in the structured inference network
    st_net_n_layers : int
        Number of hidden layers in the structured inference network
    """

    def __init__(self, st_net_hidden_dim: int, st_net_n_layers: int):
        latent_dim = 1
        obs_dim = 1
        emission_mean = lambda latent: 0.5 * latent
        emission_log_var = lambda latent: torch.log(torch.ones_like(latent) * 20).to(
            latent.device
        )
        emission_model = EmissionNormalBase(emission_mean, emission_log_var)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, st_net_hidden_dim, st_net_n_layers
        )

        transition_mean = lambda latent: latent + 0.05
        transition_log_var = lambda latent: torch.log(
            torch.ones_like(latent).to(latent.device) * 10
        )
        transition_model = DeterministicTransitionFunction(
            transition_mean, transition_log_var
        )
        super().__init__(inference_model, emission_model, transition_model)


class DMMContinuousFixedEmission(StateSpaceModel):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Emission model are fixed to the ground truth distribution
    Unlike the paper, here the transition model is not fixed and learned instead

    Parameters
    ----------
    st_net_hidden_dim : int
        Dimension of the hidden layers in the structured inference network
    st_net_n_layers : int
        Number of hidden layers in the structured inference network
    transition_hidden_dim : int
        Dimension of the hidden layers in the transition networks
    """

    def __init__(
        self, st_net_hidden_dim: int, st_net_n_layers: int, transition_hidden_dim: int
    ):
        latent_dim = 1
        obs_dim = 1
        emission_mean = lambda latent: 0.5 * latent
        emission_log_var = lambda latent: torch.log(torch.ones_like(latent) * 20).to(
            latent.device
        )
        emission_model = EmissionNormalBase(emission_mean, emission_log_var)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, st_net_hidden_dim, st_net_n_layers
        )

        transition_model = GatedTransitionFunction(latent_dim, transition_hidden_dim)
        super().__init__(inference_model, emission_model, transition_model)


class TransformerDMMContinuousFixedEmission(StateSpaceModel):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Emission model are fixed to the ground truth distribution
    Unlike the paper, here the transition model is not fixed and learned instead

    Parameters
    ----------
    nhead : int
        Number of heads in the transformer
    transition_hidden_dim : int
        Dimension of the hidden layers in the transition networks
    """

    def __init__(self, nhead: int = 1, transition_hidden_dim: int = 10):
        latent_dim = 1
        obs_dim = 1
        emission_mean = lambda latent: 0.5 * latent
        emission_log_var = lambda latent: torch.log(torch.ones_like(latent) * 20).to(
            latent.device
        )
        emission_model = EmissionNormalBase(emission_mean, emission_log_var)

        inference_model = TransformerSTLR(latent_dim, obs_dim, nhead)

        transition_model = GatedTransitionFunction(latent_dim, transition_hidden_dim)
        super().__init__(inference_model, emission_model, transition_model)


class DMMContinuous(StateSpaceModel):
    """
    DMM used for various continuous valued observation experiements

    Parameters
    ----------
    st_net_hidden_dim : int
        Dimension of the hidden layers in the structured inference network
    st_net_n_layers : int
        Number of hidden layers in the structured inference network
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    transition_hidden_dim : int
        Dimension of the hidden layers in the transition networks
    emission_hidden_size : int
        Dimension of the hidden layers in the emission network
    """

    def __init__(
        self,
        st_net_hidden_dim: int,
        st_net_n_layers: int,
        latent_dim: int,
        obs_dim: int,
        transition_hidden_dim: int,
        emission_hidden_size: int,
    ):
        emission_model = EmissionNetworkNormal(
            latent_dim, obs_dim, emission_hidden_size
        )

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, st_net_hidden_dim, st_net_n_layers
        )

        transition_model = GatedTransitionFunction(latent_dim, transition_hidden_dim)
        super().__init__(inference_model, emission_model, transition_model)


class DMMNonLinearDataset(StateSpaceModel):
    """
    DMM used for the non-linear synthetic data experiment in the paper to estimate parameters in the transition function
    """

    def __init__(
        self,
        st_net_hidden_dim: int,
        st_net_n_layers: int,
    ):
        latent_dim = 2
        obs_dim = 2
        emission_mean = lambda latent: 0.5 * latent
        emission_log_var = lambda latent: torch.log(torch.ones_like(latent) * 0.1).to(
            latent.device
        )
        emission_model = EmissionNormalBase(emission_mean, emission_log_var)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, st_net_hidden_dim, st_net_n_layers
        )

        class TransitionFunction(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
                self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))
                self.tanh = nn.Tanh()

            def forward(self, latent: torch.Tensor) -> torch.Tensor:
                latent_0 = latent[:, :, 0:1]
                latent_1 = latent[:, :, 1:2]

                latent_0_next = 0.2 * latent_0 + self.tanh(self.alpha * latent_0)
                latent_1_next = 0.2 * latent_1 + torch.sin(self.beta * latent_0)

                next_latent = torch.cat([latent_0_next, latent_1_next], dim=2)
                log_var = torch.log(torch.ones_like(next_latent) * 0.1)
                return next_latent, log_var

            def sample(self, transition_distribution):
                latent, log_var = transition_distribution
                return latent + torch.randn_like(latent) * torch.exp(0.5 * log_var)

        transition_model = TransitionFunction()
        super().__init__(inference_model, emission_model, transition_model)


class DMMBinary(StateSpaceModel):
    """
    DMM used for various binary valued observation experiements

    Parameters
    ----------
    st_net_hidden_dim : int
        Dimension of the hidden layers in the structured inference network
    st_net_n_layers : int
        Number of hidden layers in the structured inference network
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    emission_hidden_size : int
        Dimension of the hidden layers in the emission network
    transition_hidden_size : int
        Dimension of the hidden layers in the transition networks
    """

    def __init__(
        self,
        st_net_hidden_dim: int,
        st_net_n_layers: int,
        latent_dim: int,
        obs_dim: int,
        emission_hidden_size: int,
        transition_hidden_size: int,
    ):
        emission_model = EmissionNetworkBinary(
            latent_dim, obs_dim, emission_hidden_size
        )

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, st_net_hidden_dim, st_net_n_layers
        )

        transition_model = GatedTransitionFunction(latent_dim, transition_hidden_size)
        super().__init__(inference_model, emission_model, transition_model)
