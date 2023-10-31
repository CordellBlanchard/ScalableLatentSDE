"""
Transition models, used to predict the distribution of the latent variables z_t given z_t-1
Some models can only generate samples from z_t instead of the full distribution
"""
from typing import Tuple, Callable

import torch
from torch import nn
import torch.nn.functional as F


class DeterministicTransitionFunction(nn.Module):
    """
    Deterministic Transition Function for when the latent variables are deterministic
    and the transition model is known

    Parameters
    ----------
    transition_mean : Callable[[torch.Tensor], torch.Tensor]
        Function that takes in the latent variables at time step t-1 and returns
        the mean of the distribution for latent variables at time step t
    transition_sigma : Callable[[torch.Tensor], torch.Tensor]
        Function that takes in the latent variables at time step t-1 and returns
        the standard deviation of the distribution for latent variables at time step t
    """

    def __init__(
        self,
        transition_mean: Callable[[torch.Tensor], torch.Tensor],
        transition_sigma: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.transition_mean = transition_mean
        self.transition_sigma = transition_sigma

    def forward(self, prev_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model

        Parameters
        ----------
        prev_latents : torch.Tensor
            Latent variables at time step t-1, shape = (*, latent_dim)

        Returns
        -------
        transition_mean: torch.Tensor
            Mean of the distribution for latent variables at time step t, shape = (*, latent_dim)
        transition_sigma: torch.Tensor
            Standard deviation of the distribution for latent variables at time step t,
            shape = (*, latent_dim)
        """
        return self.transition_mean(prev_latents), self.transition_sigma(prev_latents)

    def sample(
        self, transition_distribution: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Sample from the distribution of the latent variables

        Parameters
        ----------
        transition_distribution : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the mean and standard deviation of the
            distribution of the latent variables

        Returns
        -------
        torch.Tensor
            Sampled latent variables, shape = (*, latent_dim)
        """
        return transition_distribution[0] + transition_distribution[
            1
        ] * torch.randn_like(transition_distribution[1])


class GatedTransitionFunction(nn.Module):
    """
    Gated Transition Function from Section 5 of the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    hidden_size: int, optional
        Dimension of the hidden layer
    """

    def __init__(self, latent_dim: int, hidden_size: int = None):
        super().__init__()
        self.latent_dim = latent_dim

        if hidden_size is None:
            hidden_size = latent_dim

        self.W_1g = nn.Linear(latent_dim, hidden_size)
        self.W_2g = nn.Linear(hidden_size, latent_dim)
        self.W_1h = nn.Linear(latent_dim, hidden_size)
        self.W_2h = nn.Linear(hidden_size, latent_dim)

        self.W_mp = nn.Linear(latent_dim, latent_dim)
        self.W_sp = nn.Linear(latent_dim, latent_dim)

        self.softplus = nn.Softplus()
        self.init_w_mp()

    def forward(self, prev_latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gated Transition Function

        Parameters
        ----------
        prev_latents : torch.Tensor
            z_t-1 latent variable, shape = (*, latent_dim)

        Returns
        -------
        m_t: torch.Tensor
            Mean of the distribution for z_t, shape = (*, latent_dim)
        s_t: torch.Tensor
            Standard deviation of the distribution for z_t, shape = (*, latent_dim)
        """
        g_t = F.sigmoid(self.W_2g(F.relu(self.W_1g(prev_latents))))
        h_t = self.W_2h(F.relu(self.W_1h(prev_latents)))
        m_t = (1 - g_t) * (self.W_mp(prev_latents)) + g_t * h_t
        s_t = self.softplus(self.W_sp(F.relu(h_t)))

        return m_t, s_t

    def init_w_mp(self) -> None:
        """
        Initialize the weights of the linear layer W_mp to be the identity matrix
        and the bias to be zero. As mentioned in the paper
        """
        self.W_mp.weight.data.copy_(torch.eye(self.latent_dim))
        self.W_mp.bias.data.fill_(0)

    def sample(
        self, transition_distribution: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Sample from the distribution of the latent variables

        Parameters
        ----------
        transition_distribution : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing the mean and standard deviation of the
            distribution of the latent variables

        Returns
        -------
        torch.Tensor
            Sampled latent variables, shape = (*, latent_dim)
        """
        return transition_distribution[0] + transition_distribution[
            1
        ] * torch.randn_like(transition_distribution[1])
