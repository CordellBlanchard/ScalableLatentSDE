"""
SDE based transition models
"""
from typing import Callable, Any, Union, List

import torch
from torch import nn


def sdeint(
    model: Callable[Any, Any],
    x0: torch.Tensor,
    ts: Union[torch.Tensor, float],
    dt: Union[torch.Tensor, float],
    n_steps: int,
) -> List[torch.Tensor]:
    """
    Integrate an SDE using Euler-Maruyama method

    Parameters
    ----------
    model : Callable[Any, Any]
        Model with f and g functions (drift and diffusion respectively)
    x0 : torch.Tensor
        Initial value of the SDE
        shape = (*, latent_dim)
    ts : Union[torch.Tensor, float]
        Start time, if tensor shape = x0.shape
    dt : Union[torch.Tensor, float]
        Time step size, if tensor shape = x0.shape
    n_steps : int
        Number of steps to integrate

    Returns
    -------
    List[torch.Tensor]
        List of values of the SDE at each time step
    """
    device = x0.device
    if isinstance(dt, float) or isinstance(dt, int):
        dt = dt * torch.ones(x0.shape).float().to(device)
    if isinstance(ts, float) or isinstance(ts, int):
        cur_t = ts * torch.ones(x0.shape).float().to(device)
    else:
        cur_t = ts
    zero_mean = torch.zeros(dt.shape).to(device)
    brownian_noise_std = torch.sqrt(dt).to(device)

    vals = [x0]
    drifts = []
    diffusions = []
    for _ in range(n_steps):
        brownian_dw = torch.normal(zero_mean, brownian_noise_std).to(device)
        drifts.append(model.f(cur_t, vals[-1]))
        diffusions.append(model.g(cur_t, vals[-1]))
        next_val = vals[-1] + drifts[-1] * dt + diffusions[-1] * brownian_dw
        vals.append(next_val)
        cur_t += dt
    return vals, drifts, diffusions


class SDETransitionTimeIndep(nn.Module):
    """
    SDE based transition model with time independent drift and diffusion functions
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int = None,
        n_euler_steps: int = 10,
    ):
        super().__init__()
        # config
        self.n_euler_steps = n_euler_steps

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size or latent_dim

        self.mu = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.sigma = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
            nn.Softplus(),
        )

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
        return sdeint(self, latents, start_times, dt, self.n_euler_steps)

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Drift function of the SDE

        Parameters
        ----------
        t : torch.Tensor
            Time
        z : torch.Tensor
            Latent variable
        """
        return self.mu(z)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function of the SDE

        Parameters
        ----------
        t : torch.Tensor
            Time
        z : torch.Tensor
            Latent variable
        """
        return self.sigma(z) + 1e-2

    def sample(self, latents: torch.Tensor) -> torch.Tensor:
        """
        The SDEs already return samples in the forward function, so sampling doesn't change the values

        Parameters
        ----------
        latents : torch.Tensor
            Latent variables, shape = (*, latent_dim)

        Returns
        -------
        torch.Tensor
            Latent variables, shape = (*, latent_dim)
        """
        return latents[0][-1]


class SDETransitionTimeDep(SDETransitionTimeIndep):
    """
    SDE based transition model with time dependent drift and diffusion functions
    """

    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(latent_dim + 1, **kwargs)

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Drift function of the SDE
        """
        f_input = torch.cat([t.unsqueeze(-1), z], dim=-1)
        return self.mu(f_input)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Diffusion function of the SDE
        """
        f_input = torch.cat([t.unsqueeze(-1), z], dim=-1)
        return self.sigma(f_input)
