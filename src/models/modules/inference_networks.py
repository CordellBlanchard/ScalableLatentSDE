"""
Implemented of various inference networks
Used to infer the latent variables from the observations
"""
from typing import Tuple

import torch
from torch import nn


class StructuredInferenceLR(nn.Module):
    """
    Structured Inference Network (ST-LR) from Section 4 of the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)

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

        self.lstm_forward = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            proj_size=latent_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm_backward = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            proj_size=latent_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.combine_layer = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh())
        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.log_var_layer = nn.Linear(latent_dim, latent_dim)

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
        z_mus : torch.Tensor
            Mean for the latent variables, shape = (batch_size, n_time_steps, latent_dim)
        z_log_vars : torch.Tensor
            Standard deviation for the latent variables,
            shape = (batch_size, n_time_steps, latent_dim)
        z_samples : torch.Tensor
            Sampled latent variables, shape = (batch_size, n_time_steps, latent_dim)

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

        batch_size, n_time_steps, obs_dim = observations.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim to be {self.obs_dim}, got {obs_dim}")

        lstm_out_left, _ = self.lstm_forward(self.linear_embedding(observations))
        lstm_out_right, _ = self.lstm_backward(
            self.linear_embedding(torch.flip(observations, [1]))
        )
        lstm_out_right = torch.flip(lstm_out_right, [1])

        # Predicting distributions for z and sampling z from these distributions
        z_mus = []
        z_log_vars = []
        z_samples = [
            torch.zeros(batch_size, 1, self.latent_dim).to(observations.device)
        ]

        for t in range(n_time_steps):
            # Combiner layer
            h_combined = (
                self.combine_layer(z_samples[-1].reshape(batch_size, self.latent_dim))
                + lstm_out_left[:, t, :]
                + lstm_out_right[:, t, :]
            ) / 3
            z_u = self.mu_layer(h_combined)
            z_log_var = self.log_var_layer(h_combined)

            # Sample z in a gradient friendly way (reparametrization trick)
            curr_z = z_u + torch.randn_like(z_log_var) * torch.exp(0.5 * z_log_var)
            z_samples.append(curr_z.reshape(batch_size, 1, self.latent_dim))

            # Save mu, sigma, sample
            z_mus.append(z_u.reshape(batch_size, 1, self.latent_dim))
            z_log_vars.append(z_log_var.reshape(batch_size, 1, self.latent_dim))

        z_mus = torch.cat(z_mus, dim=1)
        z_log_vars = torch.cat(z_log_vars, dim=1)
        z_samples = torch.cat(z_samples[1:], dim=1)

        return (z_mus, z_log_vars), z_samples


class CustomTransformer(nn.Module):
    """
    Custom Transformer module with a linear layer between the encoder and decoder

    Parameters
    ----------
    input_dim: int
        Dimension of the input
    output_dim: int
        Dimension of the output
    nhead: int
        Number of attention heads
    """

    def __init__(self, input_dim: int, output_dim: int, nhead: int):
        super(CustomTransformer, self).__init__()
        self.output_dim = output_dim

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=4
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim, nhead=nhead, dim_feedforward=16, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=1
        )

        # Linear Layer to match the output dimensionality
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Tranformer forward pass

        Parameters
        ----------
        src : torch.Tensor
            Input, shape = (batch_size, n_time_steps, input_dim)

        Returns
        -------
        torch.Tensor
            Output, shape = (batch_size, n_time_steps, output_dim)
        """
        batch_size, n_timesteps, _ = src.shape
        memory = self.transformer_encoder(src)

        reshaped_memory = self.fc(memory.reshape(-1, src.shape[-1])).reshape(
            batch_size, n_timesteps, -1
        )

        tgt = torch.zeros((batch_size, n_timesteps + 1, self.output_dim)).to(src.device)
        for t in range(n_timesteps):
            output = self.transformer_decoder(tgt, reshaped_memory)
            tgt[:, t + 1] = output[:, t]

        tgt = tgt[:, 1:]
        return tgt


class TransformerSTLR(nn.Module):
    """
    Structured Inference Network (ST-LR) from Section 4 of the paper

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    """

    def __init__(self, latent_dim: int, obs_dim: int, nhead: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        self.transformer = CustomTransformer(obs_dim, latent_dim * 2, nhead)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ST-LR

        Parameters
        ----------
        x : torch.Tensor
            Observations, shape = (batch_size, n_time_steps, obs_dim)

        Returns
        -------
        z_mus : torch.Tensor
            Mean for the latent variables, shape = (batch_size, n_time_steps, latent_dim)
        z_log_vars : torch.Tensor
            Standard deviation for the latent variables,
            shape = (batch_size, n_time_steps, latent_dim)
        z_samples : torch.Tensor
            Sampled latent variables, shape = (batch_size, n_time_steps, latent_dim)

        Raises
        ------
        ValueError
            If x does not have 3 dimensions (batch_size, n_time_steps, obs_dim)
            If obs_dim is not equal to self.obs_dim
        """
        if len(x.shape) != 3:
            raise ValueError(f"Expected x to have 3 dimensions, got {len(x.shape)}")

        batch_size, n_time_steps, obs_dim = x.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim to be {self.obs_dim}, got {obs_dim}")

        tgt = self.transformer(x)
        tgt = tgt.reshape(batch_size, n_time_steps, 2, self.latent_dim)

        # gather z_mus and z_log_vars from transformer output
        z_mus = tgt[:, :, 0, :]
        z_log_vars = tgt[:, :, 1, :]
        z_samples = z_mus + torch.normal(0, 1, z_log_vars.shape).to(
            z_mus.device
        ) * torch.exp(0.5 * z_log_vars)

        return [z_mus, z_log_vars], z_samples
