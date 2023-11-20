"""
General Synthetic State Space Model Dataset
"""
from typing import Callable, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class GSSM(Dataset):
    """
    Generative State Space Model Dataset

    Parameters
    ----------
    transition_func: Callable[[np.array, int], np.array]
        Function that takes in the previous latent state and the current time step
        and returns the next latent state. Input and output shape is (n_samples, n_latent)
    emission_func: Callable[[np.array, int], np.array]
        Function that takes in the current latent state and the current time step
        and returns the current observation. Input and output shape is (n_samples, n_obs)
    gen_initial_latent: Callable[[int], np.array]
        Function that takes in the number of samples and returns the initial latent state.
        Output shape is (n_samples, n_latent)
    n_samples: int
        Number of samples to generate
    n_time_steps: int
        Number of time steps to generate
    return_latents: bool
        Whether to return the latents
    return_times: bool
        Whether to return the time steps

    Attributes
    ----------
    observations: torch.Tensor
        Generated observations of shape (n_samples, n_time_steps, n_obs)
    latent_variables: torch.Tensor
        Generated latent variables of shape (n_samples, n_time_steps, n_latent)
    """

    def __init__(
        self,
        transition_func: Callable[[np.array, int], np.array],
        emission_func: Callable[[np.array, int], np.array],
        gen_initial_latent: Callable[[int], np.array],
        n_samples: int = 5000,
        n_time_steps: int = 25,
        return_latents: bool = False,
        return_times: bool = False,
    ):
        self.transition_func = transition_func
        self.emission_func = emission_func
        self.generate_initial_latent = gen_initial_latent
        self.n_samples = n_samples
        self.n_time_steps = n_time_steps
        self.return_latents = return_latents
        self.return_times = return_times

        # Generate data using the given functions
        observations, latent_variables = self.generate_data()
        self.observations = torch.from_numpy(observations).float()
        self.latent_variables = torch.from_numpy(latent_variables).float()

        if return_times:
            self.times = torch.from_numpy(np.arange(n_time_steps)).float()

    def generate_data(self) -> Tuple[np.array, np.array]:
        """
        Generate data using the given transition, emission, and initial latent functions

        Returns
        -------
        observations: np.array
            Generated observations of shape (n_samples, n_time_steps, n_obs)
        latent_variables: np.array
            Generated latent variables of shape (n_samples, n_time_steps, n_latent)
        """
        observations = []
        latents = []
        for t in range(self.n_time_steps):
            if t == 0:
                latents.append(self.generate_initial_latent(self.n_samples))
            else:
                latents.append(self.transition_func(latents[t - 1], t))
            observations.append(self.emission_func(latents[t], t))
        observations, latents = np.array(observations), np.array(latents)

        # Transpose in order to have the dimensions be (n_samples, n_time_steps, n_obs/latent)
        observations = np.transpose(observations, (1, 0, 2))
        latents = np.transpose(latents, (1, 0, 2))

        return observations, latents

    def plot(self, n_samples: int = 5, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Plot a few samples of latent variables and observations from the dataset
        Each dimensions is plotted separately

        Parameters
        ----------
        n_samples : int, by default 5
            Number of samples to plot
        """
        samples_indices = np.random.choice(self.n_samples, n_samples, replace=False)
        latents = self.latent_variables.numpy()[
            samples_indices
        ]  # shape = (n_samples, n_time_steps, latent_dim)
        observations = self.observations.numpy()[
            samples_indices
        ]  # shape = (n_samples, n_time_steps, obs_dim)

        # Plot for each latent dimension
        for i in range(latents.shape[2]):
            plt.figure(figsize=figsize)
            plt.title(f"Latent Dimension {i}")
            for j in range(latents.shape[0]):
                plt.plot(latents[j, :, i])
            plt.show()

        # Plot for each observation dimension
        for i in range(observations.shape[2]):
            plt.figure(figsize=figsize)
            plt.title(f"Observation Dimension {i}")
            for j in range(observations.shape[0]):
                plt.plot(observations[j, :, i])
            plt.show()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        obs = [self.observations[idx]]
        if self.return_latents:
            obs.append(self.latent_variables[idx])

        if self.return_times:
            obs.append(self.times)
        return obs
