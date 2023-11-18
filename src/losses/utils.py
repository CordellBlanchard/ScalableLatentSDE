"""
Helpers function for losses
"""
from typing import Tuple
import numpy as np
import torch


def gaussian_nll(
    mu: torch.Tensor, log_var: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
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
    return 0.5 * (np.log(2 * np.pi) + log_var + (x - mu) ** 2 / torch.exp(log_var))


def gaussian_kl(
    mu_q: torch.Tensor,
    log_cov_q: torch.Tensor,
    mu_prior: torch.Tensor,
    log_cov_prior: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence between two Gaussians
    From theano/models/__init__.py

    Parameters
    ----------
    mu_q : torch.Tensor
        Mean of the first Gaussian
    log_cov_q : torch.Tensor
        Log covariance of the first Gaussian
    mu_prior : torch.Tensor
        Mean of the second Gaussian
    log_cov_prior : torch.Tensor
        Log covariance of the second Gaussian

    Returns
    -------
    torch.Tensor
        KL divergence between the two Gaussians
    """
    diff_mu = mu_prior - mu_q
    cov_q = torch.exp(log_cov_q)
    cov_prior = torch.exp(log_cov_prior)
    KL = log_cov_prior - log_cov_q - 1.0 + cov_q / cov_prior + diff_mu**2 / cov_prior
    KL_t = 0.5 * KL
    return KL_t


def impute_transition_distribution(
    transition_distribution: Tuple[torch.Tensor, torch.Tensor],
    mean: float,
    log_var: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Impute the first time step of the transition distribution

    Parameters
    ----------
    transition_distribution : Tuple[torch.Tensor, torch.Tensor]
        Mean and log variance of the transition distribution
    mean : float
        Mean to impute
    log_var : float
        Log variance to impute

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Imputed transition distribution
    """
    transition_distribution = (
        torch.cat(
            [
                torch.zeros_like(transition_distribution[0][:, 0:1, :]) + mean,
                transition_distribution[0][:, :-1],
            ],
            dim=1,
        ),
        torch.cat(
            [
                torch.zeros_like(transition_distribution[1][:, 0:1, :]) + log_var,
                transition_distribution[1][:, :-1],
            ],
            dim=1,
        ),
    )
    return transition_distribution
