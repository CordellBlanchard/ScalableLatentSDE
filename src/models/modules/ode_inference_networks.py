"""
Implemented of various inference networks
Used to infer the latent variables from the observations
"""
from typing import Tuple

import torch
from torch import nn


class RNNInferenceNetwork(nn.Module):
    """
    RNN Inference network for latent ODE

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    hidden_dim : int
        Dimension of the hidden layer
    n_layers : int
        Number of layers in the LSTM
    """

    def __init__(
        self, latent_dim: int, obs_dim: int, hidden_dim: int = 256, n_layers: int = 8
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.linear_embedding = nn.Linear(obs_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            proj_size=latent_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.g = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(
        self, observations: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        ST-LR forward pass

        Parameters
        ----------
        observations : torch.Tensor
            Observations, shape = (batch_size, n_time_steps, obs_dim)

        Returns
        -------
        z_0_mu : torch.Tensor
            Mean for z0, shape = (batch_size, latent_dim)
        z0_log_var : torch.Tensor
            Log var for z0, shape = (batch_size, latent_dim)

        Raises
        ------
        ValueError
            If observations does not have 3 dimensions (batch_size, n_time_steps, obs_dim)
            If obs_dim is not equal to self.obs_dim
        """
        if len(observations.shape) != 3:
            raise ValueError(
                f"Expected observations to have 3 dimensions, got {len(observations.shape)}"
            )

        batch_size, _, obs_dim = observations.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim to be {self.obs_dim}, got {obs_dim}")

        lstm_out, _ = self.lstm(self.linear_embedding(torch.flip(observations, [1])))

        lstm_out = lstm_out[:, -1, :]
        g_out = self.g(lstm_out).reshape(batch_size, self.latent_dim, 2)
        return g_out[:, :, 0], g_out[:, :, 1]
