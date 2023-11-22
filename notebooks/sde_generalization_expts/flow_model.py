"""
Flow model for density estimation based on RealNVP
"""
from typing import Tuple

import torch
from torch import nn


class FlowLayer(nn.Module):
    """
    A single layer of a flow model. This is a simple implementation of the
    RealNVP model from the paper "Density estimation using Real NVP" by
    Dinh et al. (2016)

    Parameters
    ----------
    hidden_dim : int
        The number of hidden units in the neural network used to compute the
        parameters of the affine transformation
    """

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.log_s1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_s2 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.t1 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.t2 = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the flow layer

        Parameters
        ----------
        x : torch.Tensor
            Input samples, shape = (batch_size, 2)

        Returns
        -------
        x : torch.Tensor
            Transformed samples, shape = (batch_size, 2)
        log_jacobian : torch.Tensor
            Log of the determinant of the Jacobian of the transformation
            shape = (batch_size,)
        """
        x0 = x[:, 0:1]
        x1 = x[:, 1:]

        log_s1 = self.log_s1(x0)
        t1 = self.t1(x0)

        x1 = torch.exp(log_s1) * x1 + t1

        log_s2 = self.log_s2(x1)
        t2 = self.t2(x1)

        x0 = torch.exp(log_s2) * x0 + t2

        x = torch.cat([x0, x1], dim=1)

        log_jacobian = log_s1 + log_s2

        return x, log_jacobian[:, 0]

    def reverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse pass of the flow layer

        Parameters
        ----------
        x : torch.Tensor
            Input samples, shape = (batch_size, 2)

        Returns
        -------
        x : torch.Tensor
            Transformed samples, shape = (batch_size, 2)
        log_jacobian : torch.Tensor
            Log of the determinant of the Jacobian of the transformation
            shape = (batch_size,)
        """
        x0 = x[:, 0:1]
        x1 = x[:, 1:]

        log_s2 = self.log_s2(x1)
        t2 = self.t2(x1)

        x0 = (x0 - t2) / (torch.exp(log_s2) + 1e-3)

        log_s1 = self.log_s1(x0)
        t1 = self.t1(x0)
        x1 = (x1 - t1) / (torch.exp(log_s1) + 1e-3)

        x = torch.cat([x0, x1], dim=1)

        log_jacobian = -1 * (log_s1 + log_s2)

        return x, log_jacobian[:, 0]


class Flow(nn.Module):
    """
    A flow model consisting of multiple flow layers

    Parameters
    ----------
    n_layers : int
        The number of flow layers in the model
    hidden_dim : int
        The number of hidden units in the neural network used to compute the
        parameters of the affine transformation
    """

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.layers = nn.Sequential(*[FlowLayer(hidden_dim)] * n_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the flow layer

        Parameters
        ----------
        x : torch.Tensor
            Input samples, shape = (batch_size, 2)

        Returns
        -------
        x : torch.Tensor
            Transformed samples, shape = (batch_size, 2)
        log_jacobian : torch.Tensor
            Log of the determinant of the Jacobian of the network
            shape = (batch_size,)
        """
        total_log_jacobian = 0

        for layer in self.layers:
            x, log_jacobian = layer.forward(x)
            total_log_jacobian += log_jacobian

        return x, total_log_jacobian

    def reverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse pass of the flow network

        Parameters
        ----------
        x : torch.Tensor
            Input samples, shape = (batch_size, 2)

        Returns
        -------
        x : torch.Tensor
            Transformed samples, shape = (batch_size, 2)
        log_jacobian : torch.Tensor
            Log of the determinant of the Jacobian of the network
            shape = (batch_size,)
        """
        total_log_jacobian = 0

        for layer in self.layers[::-1]:
            x, log_jacobian = layer.reverse(x)
            total_log_jacobian += log_jacobian

        return x, total_log_jacobian
