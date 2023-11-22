"""
Utils for plotting results
"""
import os
import sys

notebook_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(os.path.dirname(notebook_dir))
sys.path.append(parent_dir)
from src.models import *

import yaml
from typing import Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_synthetic_dataset(
    observations: torch.Tensor,
    latents: torch.Tensor,
    models: Tuple[torch.nn.Module, torch.nn.Module, Any],
) -> None:
    """
    Takes as input the observations, latents and models and plots the results.

    Parameters
    ----------
    observations : torch.Tensor
        Tensor of observations, shape = (n_observations, n_time_steps, obs_dimension)
    latents : torch.Tensor
        Tensor of latents, shape = (n_observations, n_time_steps, latent_dimension)
    models : Tuple[torch.nn.Module, torch.nn.Module, Any]
        Tuple of models, (dmm, sde, ari)
    """
    dmm = models[0]
    latent_distribution, _ = dmm.inference_model(observations)
    emission_distribution = dmm.emission_model(latent_distribution[0])
    dmm_latent_means, dmm_latent_log_var = latent_distribution
    dmm_emission_means, dmm_emission_log_var = emission_distribution

    sde = models[1]
    latent_distribution, _ = sde.inference_model(observations)
    emission_distribution = sde.emission_model(latent_distribution[0])
    sde_latent_means, sde_latent_log_var = latent_distribution
    sde_emission_means, sde_emission_log_var = emission_distribution

    ode = models[2]
    z0_mu, z0_log_var = ode.inference_model(observations)
    z0 = z0_mu + torch.randn_like(z0_mu) * torch.exp(0.5 * z0_log_var)
    transition_samples = ode.transition_model(z0, 24, 0.1)
    emission_distribution = ode.emission_model(transition_samples)
    ode_emission_means, ode_emission_log_var = emission_distribution
    ode_latent_means, ode_latent_log_var = transition_samples, torch.zeros(
        transition_samples.shape
    )

    ari = models[3]
    dataset = (
        observations.detach().numpy()
    )  # (n_observations, n_time_steps, dimensions)
    predictions = [dataset[:, 0, 0], dataset[:, 1, 0]]
    for i in range(dataset.shape[1] - 2):
        prediction = ari.predict_future(
            test_data=dataset[:, : i + 2, 0], steps=1, diff_order=[1], coef=ari.coef
        )
        predictions.append(prediction.reshape(-1))
    predictions = np.array(predictions).T

    i = 0  # sample index to plot

    # plot latent with confidence interval
    plt.title("Latents")
    plt.plot(latents[i, :, 0].numpy(), color="blue", label="Ground Truth")
    plt.plot(dmm_latent_means[i, :, 0].detach().numpy(), color="orange", label="DMM")
    plt.fill_between(
        np.arange(dmm_latent_means.shape[1]),
        dmm_latent_means[i, :, 0].detach().numpy()
        - 2 * torch.exp(0.5 * dmm_latent_log_var[i, :, 0]).detach().numpy(),
        dmm_latent_means[i, :, 0].detach().numpy()
        + 2 * torch.exp(0.5 * dmm_latent_log_var[i, :, 0]).detach().numpy(),
        alpha=0.2,
        color="orange",
    )
    plt.plot(sde_latent_means[i, :, 0].detach().numpy(), label="SDE")
    plt.fill_between(
        np.arange(sde_latent_means.shape[1]),
        sde_latent_means[i, :, 0].detach().numpy()
        - 2 * torch.exp(0.5 * sde_latent_log_var[i, :, 0]).detach().numpy(),
        sde_latent_means[i, :, 0].detach().numpy()
        + 2 * torch.exp(0.5 * sde_latent_log_var[i, :, 0]).detach().numpy(),
        alpha=0.2,
        color="green",
    )
    plt.plot(ode_latent_means[i, :, 0].detach().numpy(), color="black", label="ODE")
    plt.fill_between(
        np.arange(ode_latent_means.shape[1]),
        ode_latent_means[i, :, 0].detach().numpy()
        - 2 * torch.exp(0.5 * ode_latent_log_var[i, :, 0]).detach().numpy(),
        ode_latent_means[i, :, 0].detach().numpy()
        + 2 * torch.exp(0.5 * ode_latent_log_var[i, :, 0]).detach().numpy(),
        alpha=0.2,
        color="black",
    )
    plt.legend()
    plt.grid()
    plt.show()

    plt.title("Observations")
    plt.plot(observations[i, :, 0].detach().numpy(), color="blue", label="Ground Truth")
    plt.plot(dmm_emission_means[i, :, 0].detach().numpy(), color="orange", label="DMM")
    plt.fill_between(
        np.arange(dmm_emission_means.shape[1]),
        dmm_emission_means[i, :, 0].detach().numpy()
        - 2 * torch.exp(0.5 * dmm_emission_log_var[i, :, 0]).detach().numpy(),
        dmm_emission_means[i, :, 0].detach().numpy()
        + 2 * torch.exp(0.5 * dmm_emission_log_var[i, :, 0]).detach().numpy(),
        alpha=0.2,
        color="orange",
    )
    plt.plot(sde_emission_means[i, :, 0].detach().numpy(), color="green", label="SDE")
    plt.fill_between(
        np.arange(sde_emission_means.shape[1]),
        sde_emission_means[i, :, 0].detach().numpy()
        - 2 * torch.exp(0.5 * sde_emission_log_var[i, :, 0]).detach().numpy(),
        sde_emission_means[i, :, 0].detach().numpy()
        + 2 * torch.exp(0.5 * sde_emission_log_var[i, :, 0]).detach().numpy(),
        alpha=0.2,
        color="green",
    )
    plt.plot(ode_emission_means[i, :, 0].detach().numpy(), color="black", label="ODE")
    plt.fill_between(
        np.arange(ode_emission_means.shape[1]),
        ode_emission_means[i, :, 0].detach().numpy()
        - 2 * torch.exp(0.5 * ode_emission_log_var[i, :, 0]).detach().numpy(),
        ode_emission_means[i, :, 0].detach().numpy()
        + 2 * torch.exp(0.5 * ode_emission_log_var[i, :, 0]).detach().numpy(),
        alpha=0.2,
        color="black",
    )
    plt.plot(predictions[i], label="ARI", color="red")
    plt.legend()
    plt.grid()
    plt.show()


def load_model_from_config(config_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model from config file

    Parameters
    ----------
    config_path : str
        Path to yaml config file
    device : torch.device
        Device to load model on

    Returns
    -------
    torch.nn.Module
        Model loaded from config file
    """
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    model = eval(config["model_name"])(**config["model_params"])
    model_path = os.path.join("../../", config["trainer_params"]["save_path"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model
