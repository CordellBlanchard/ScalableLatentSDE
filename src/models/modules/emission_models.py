"""
Implementation of various emission models
Used to generate observations from the latent variables
"""
from typing import Tuple, Callable

import torch
from torch import nn


class EmissionNormalBase(nn.Module):
    """
    Emission network base class for continuous valued observations

    Parameters
    ----------
    emission_mean_func : Callable[[torch.Tensor], torch.Tensor]
        Function that maps latents to emission means
    emission_sigma_func : Callable[[torch.Tensor], torch.Tensor]
        Function that maps latents to emission standard deviations
    """

    def __init__(
        self,
        emission_mean_func: Callable[[torch.Tensor], torch.Tensor],
        emission_sigma_func: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.emission_mean_func = emission_mean_func
        self.emission_sigma_func = emission_sigma_func

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Emission Network forward pass

        Parameters
        ----------
        latents : torch.Tensor
            Latent variable, shape = (*, latent_dim)

        Returns
        -------
        obs_mean: torch.Tensor
            Mean of the distribution for observations, shape = (*, obs_dim)
        obs_sigma: torch.Tensor
            Standard deviation of the distribution for observations, shape = (*, obs_dim)
        """
        return self.emission_mean_func(latents), self.emission_sigma_func(latents)

    def sample(
        self, emission_distribution: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Sample from the emission distribution

        Parameters
        ----------
        emission_distribution : Tuple[torch.Tensor, torch.Tensor]
            Tuple of emission mean and standard deviation

        Returns
        -------
        torch.Tensor
            Sample from the emission distribution, same shape as each of the inputs
        """
        return emission_distribution[0] + emission_distribution[1] * torch.randn_like(
            emission_distribution[1]
        )


class EmissionBinaryBase(nn.Module):
    """
    Emission network base class for binary valued observations

    Parameters
    ----------
    emission_prob_func : Callable[[torch.Tensor], torch.Tensor]
        Function that maps latents to emission probabilities
    """

    def __init__(self, emission_prob_func: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.emission_prob_func = emission_prob_func

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Emission Network
        """
        return self.emission_prob_func(latents)

    def sample(self, emission_distribution: torch.Tensor) -> torch.Tensor:
        """
        Sample from the emission distribution
        Note: these samples don't support gradients

        Parameters
        ----------
        emission_distribution : torch.Tensor
            Probability of observations being =1, shape = (*, obs_dim)

        Returns
        -------
        torch.Tensor
            Sample from the emission distribution, same shape as each of the inputs
        """
        return (
            torch.rand(emission_distribution.shape).to(emission_distribution.device)
            < emission_distribution
        )


class EmissionNetworkNormal(EmissionNormalBase):
    """
    Emission Network for continuous valued observations, inspired by Section 5 of the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    hidden_size: int, optional
        Dimension of the hidden layer
    """

    def __init__(self, latent_dim: int, obs_dim: int, hidden_size: int = None):
        self.latent_dim = latent_dim  # input layer size (size of z)
        self.obs_dim = obs_dim  # observation x size

        # hidden size defaults to input size if not given
        if hidden_size is None:
            hidden_size = self.latent_dim

        mu_net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.obs_dim),
        )
        sigma_net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.obs_dim),
            nn.Softplus(),
        )

        super().__init__(mu_net, sigma_net)


class EmissionNetworkBinary(EmissionBinaryBase):
    """
    Emission network implemented from Section 5 of the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    hidden_size: int, optional
        Dimension of the hidden layer
    """

    def __init__(self, latent_dim: int, obs_dim: int, hidden_size: int = None):
        self.latent_dim = latent_dim  # input layer size (size of z)
        self.obs_dim = obs_dim  # observation x size

        # hidden size defaults to input size if not given
        if hidden_size is None:
            hidden_size = self.latent_dim

        bin_net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.obs_dim),
            nn.Sigmoid(),
        )
        super().__init__(bin_net)
