"""
SDE based transition models
"""
from typing import Callable, Any, Union, List

import torch
from torch import nn


def odeint(
    model: Callable[Any, Any],
    x0: torch.Tensor,
    ts: Union[torch.Tensor, float],
    dt: Union[torch.Tensor, float],
    n_steps: int,
) -> List[torch.Tensor]:
    """
    Integrate an ODE using Euler method

    Parameters
    ----------
    model : Callable[Any, Any]
        Model with forward function representing the ODE
    x0 : torch.Tensor
        Initial value of the ODE
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
        List of values of the ODE at each time step
    """
    device = x0.device
    if isinstance(dt, float) or isinstance(dt, int):
        dt = dt * torch.ones(x0.shape).float().to(device)
    if isinstance(ts, float) or isinstance(ts, int):
        cur_t = ts * torch.ones(x0.shape).float().to(device)
    else:
        cur_t = ts

    vals = [x0]
    for _ in range(n_steps):
        drift = model.drift(vals[-1])
        next_val = vals[-1] + drift * dt
        vals.append(next_val)
        cur_t += dt
    return vals


class ODETransitionTimeIndep(nn.Module):
    """
    ODE based transition model with time independent drift
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

        self.drift = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(
        self, latents: torch.Tensor, max_time: int, dt: torch.Tensor
    ) -> torch.Tensor:
        """
        ODE integration using Euler method

        Parameters
        ----------
        latents : torch.Tensor
            Start values of the ODE, shape = (*, latent_dim)
        needed_times : torch.Tensor
            Times to get from the ODE, shape = (*, 1)
        dt : torch.Tensor
            Time step size, shape = (*, 1)

        Returns
        -------
        torch.Tensor
            Values of the ODE at the final time step, shape = (batch_size, max_time, latent_dim)
        """
        all_latents = [latents]
        for t in range(max_time):
            next_latents = odeint(self, all_latents[-1], t, dt, self.n_euler_steps)[-1]
            all_latents.append(next_latents)
        return torch.stack(all_latents, dim=1)

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


class ODEAdjoint(ODETransitionTimeIndep):
    def forward(self, t, latents):
        return self.drift(latents)
