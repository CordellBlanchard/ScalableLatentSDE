"""
Synthetic datasets from the paper:
Structured Inference Networks for Nonlinear State Space Models, Krishnan et al. 2016
"""
import numpy as np
from .gssm import GSSM


class LinearGSSM(GSSM):
    """
    Linear Generative State Space Model Dataset as in the paper, Section 6, Figure 3
    Latent state is 1-dimensional, observation is 1-dimensional

    Parameters
    ----------
    n_samples: int
        Number of samples to generate
    n_time_steps: int
        Number of time steps to generate
    return_times: bool
        Whether to return the time steps
    """

    def __init__(
        self, n_samples: int = 5000, n_time_steps: int = 25, return_times: bool = False
    ):
        std_z = np.sqrt(10)
        std_x = np.sqrt(20)

        def transition_func(prev_latent: np.array, t: int) -> np.array:
            return np.random.normal(0.05, std_z, size=prev_latent.shape) + prev_latent

        def emission_func(latent: np.array, t: int) -> np.array:
            return np.random.normal(0, std_x, size=latent.shape) + (latent * 0.5)

        def gen_initial_latent(n_samples: int) -> np.array:
            return np.random.normal(0, std_z, size=(n_samples, 1))

        super().__init__(
            transition_func,
            emission_func,
            gen_initial_latent,
            n_samples,
            n_time_steps,
            return_times,
        )


class NonLinearGSSM(GSSM):
    """
    NonLinear Generative State Space Model Dataset as in the paper, Section 6, Figure 3
    Latent state is 2-dimensional, observation is 2-dimensional

    Parameters
    ----------
    alpha: float
        Parameter for the transition function
    beta: float
        Parameter for the transition function
    n_samples: int
        Number of samples to generate
    n_time_steps: int
        Number of time steps to generate
    return_times: bool
        Whether to return the time steps
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = -0.1,
        n_samples: int = 5000,
        n_time_steps: int = 25,
        return_times: bool = False,
    ):
        def transition_func(prev_latent: np.array, t: int) -> np.array:
            z0 = prev_latent[:, 0]
            z1 = prev_latent[:, 1]
            next_z0 = 0.2 * z0 + np.tanh(alpha * z1)
            next_z1 = 0.2 * z1 + np.sin(beta * z0)
            next_z_mean = np.stack([next_z0, next_z1], axis=1)
            return np.random.normal(0, 1, size=next_z_mean.shape) + next_z_mean

        def emission_func(latent: np.array, t: int) -> np.array:
            return np.random.normal(0, 0.1, size=latent.shape) + (latent * 0.5)

        def gen_initial_latent(n_samples: int) -> np.array:
            return np.random.normal(0, 1, size=(n_samples, 2))

        super().__init__(
            transition_func,
            emission_func,
            gen_initial_latent,
            n_samples,
            n_time_steps,
            return_times,
        )
