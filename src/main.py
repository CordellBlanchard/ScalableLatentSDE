import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from model import *
from data_loaders import *

def train(x):
    dataset = SyntheticDatasetLoader(x.numpy())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model = ST_LR(latent_dim=1, obs_dim=1)
    emission_model = EmissionNetwork(input_size=1, hidden_size=10, observation_size=1)
    transition_model = GatedTransitionFunction(input_size=1, hidden_size=10)

    optimizer = torch.optim.Adam(list(inference_model.parameters()) + list(emission_model.parameters()) + list(transition_model.parameters()), lr=1e-4)

    inference_model = inference_model.to(device)
    emission_model = emission_model.to(device)
    transition_model = transition_model.to(device)
    criterion = VLBLoss()

    inference_model.train()
    emission_model.train()
    transition_model.train()

    for epoch in range(350):
        for data in tqdm(dataloader):
            data = data.to(device) # shape = (batch_size, T)

            # run inference model
            z_mus, z_sigmas, z_samples = inference_model(data.reshape(data.shape[0], data.shape[1], 1))

            # run emission model
            emission_mus, emission_sigmas = emission_model(z_samples)

            # run transition model
            transition_mus, transition_sigmas = transition_model(z_samples[:,:-1])

            loss = criterion(z_mus, z_sigmas, emission_mus, emission_sigmas, transition_mus, transition_sigmas, data).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]	Loss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(dataloader.dataset),
            #         100. * batch_idx / len(dataloader), loss.item()))
        print("Epoch ", epoch, ", loss: ", loss.item())

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Parameters
    N = 5000  # Number of sequences
    T = 25    # Sequence length
    sigma_z = torch.sqrt(torch.tensor(10.0))  # Standard deviation for z
    sigma_x = torch.sqrt(torch.tensor(20.0))  # Standard deviation for x

    # Preallocate tensors
    z = torch.zeros((N, T))
    x = torch.zeros((N, T))

    # Generation loop
    for n in range(N):
        for t in range(T):
            # Generate z_t using GSSM z_t = N(z_{t-1} + 0.05, 10)
            z[n, t] = torch.normal(mean=z[n, t-1] + 0.05, std=sigma_z) if t > 0 else torch.normal(mean=0.05, std=sigma_z)

            # Generate x_t using GSSM x_t = N(0.5 * z_t, 20)
            x[n, t] = torch.normal(mean=0.5 * z[n, t], std=sigma_x)
    
    # train our model
    train(x)