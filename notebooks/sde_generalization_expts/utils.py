from typing import Callable, Any, Union, List

import numpy as np
import torch


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


def gaussian_nll(mu: float, log_var: float, x: torch.Tensor) -> torch.Tensor:
    """
    Gaussian negative log likelihood
    From theano/models/__init__.py

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the Gaussian
    log_var : torch.Tensor
        Log variance of the Gaussian
    x: torch.Tensor
        Observation

    Returns
    -------
    torch.Tensor
        Negative log likelihood of the observation under the Gaussian
    """
    return 0.5 * (np.log(2 * np.pi) + log_var + (x - mu) ** 2 / np.exp(log_var))


def gen_data(
    x_dist_std,
    source_fn,
    target_fn,
    source_std,
    target_std,
    num_points=1000,
    batch_size=64,
):
    x = np.random.normal(0, x_dist_std, size=num_points)
    y_cos = source_fn(torch.from_numpy(x)) + np.random.normal(
        0, source_std, size=num_points
    )
    y_sin = target_fn(torch.from_numpy(x)) + np.random.normal(
        0, target_std, size=num_points
    )

    cos_data = np.vstack([x, y_cos]).T
    sin_data = np.vstack([x, y_sin]).T
    cos_loader = torch.utils.data.DataLoader(
        torch.from_numpy(cos_data).float(), batch_size=batch_size, shuffle=True
    )
    sin_loader = torch.utils.data.DataLoader(
        torch.from_numpy(sin_data).float(), batch_size=batch_size, shuffle=True
    )

    loaders = {"source": cos_loader, "target": sin_loader}
    return loaders
