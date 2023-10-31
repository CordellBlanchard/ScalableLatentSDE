"""
Implementations of the DMM for various experiements from the paper:
Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
"""
import torch

from .base import StateSpaceModel
from .modules import (
    EmissionNormalBase,
    EmissionNetworkNormal,
    StructuredInferenceLR,
    DeterministicTransitionFunction,
    GatedTransitionFunction,
)


class DMMContinuousFixedTheta(StateSpaceModel):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Both emission and transition models are fixed to the ground truth distributions
    Note: emission sigma is set to 1 instead of 20 as in the paper, found this to work better
    """

    def __init__(self):
        latent_dim = 1
        obs_dim = 1
        emission_mean = lambda latent: 0.5 * latent
        emission_sigma = lambda latent: torch.ones_like(latent).to(latent.device)
        emission_model = EmissionNormalBase(emission_mean, emission_sigma)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, hidden_dim=256, n_layers=8
        )

        transition_mean = lambda latent: latent + 0.05
        transition_sigma = lambda latent: torch.ones_like(latent).to(latent.device) * 10
        transition_model = DeterministicTransitionFunction(
            transition_mean, transition_sigma
        )
        super().__init__(inference_model, emission_model, transition_model)


class DMMContinuousFixedEmission(StateSpaceModel):
    """
    DMM used for the synthetic data experiment in the paper:
    Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. (2016)
    Specifically the linear synthetic dataset
    Emission model are fixed to the ground truth distribution
    Unlike the paper, here the transition model is not fixed and learned instead
    Note: emission sigma is set to 1 instead of 20 as in the paper, found this to work better
    """

    def __init__(self):
        latent_dim = 1
        obs_dim = 1
        emission_mean = lambda latent: 0.5 * latent
        emission_sigma = lambda latent: torch.ones_like(latent).to(latent.device)
        emission_model = EmissionNormalBase(emission_mean, emission_sigma)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, hidden_dim=256, n_layers=8
        )

        transition_model = GatedTransitionFunction(latent_dim, hidden_size=10)
        super().__init__(inference_model, emission_model, transition_model)


class DMMContinuous(StateSpaceModel):
    """
    DMM used for various continuous valued observation experiements

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    """

    def __init__(self, latent_dim: int, obs_dim: int):
        emission_model = EmissionNetworkNormal(latent_dim, obs_dim, hidden_size=10)

        inference_model = StructuredInferenceLR(
            latent_dim, obs_dim, hidden_dim=256, n_layers=8
        )

        transition_model = GatedTransitionFunction(latent_dim, hidden_size=10)
        super().__init__(inference_model, emission_model, transition_model)
