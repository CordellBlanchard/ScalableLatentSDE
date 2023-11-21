"""
Two dimensional SDE model for the SDE generalization experiments.
"""
import torch
from torch import nn
from utils import sdeint


class SDE(torch.nn.Module):
    """
    SDE model for the SDE generalization experiments.

    Parameters
    ----------
    n_euler_steps : int
        Number of Euler-Maruyama steps to use for integration
    """

    def __init__(self, n_euler_steps: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.mu2 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sigma = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sigma2 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.n_euler_steps = n_euler_steps

    # Drift
    def f(self, t, y):
        y0 = y[:, 0:1]
        mu0 = self.mu(y0)
        mu1 = self.mu2(y)
        out = torch.cat([mu0, mu1], dim=1)
        return out

    # Diffusion
    def g(self, t, y):
        y0 = y[:, 0:1]
        sigma0 = self.sigma(y0)
        sigma1 = self.sigma2(y)
        out = torch.cat([sigma0, sigma1], dim=1).reshape(y0.shape[0], 2)
        return out

    def forward(
        self, latents: torch.Tensor, start_times: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """
        SDE integration using Euler-Maruyama method

        Parameters
        ----------
        latents : torch.Tensor
            Start values of the SDE, shape = (*, latent_dim)
        start_times : torch.Tensor
            Start times of the SDE, shape = (*, 1)
        dt : torch.Tensor
            Time step size, shape = (*, 1)

        Returns
        -------
        torch.Tensor
            Values of the SDE at the final time step, shape = (*, latent_dim)
        """
        return sdeint(self, latents, start_times, dt, self.n_euler_steps)[0]
