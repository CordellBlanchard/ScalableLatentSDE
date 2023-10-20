from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

class CustomTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers):
        super(CustomTransformer, self).__init__()
        self.output_dim = output_dim

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

        # Linear Layer to match the output dimensionality
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        batch_size, n_timesteps, _ = src.shape
        memory = self.transformer_encoder(src)

        reshaped_memory = self.fc(memory.reshape(-1, src.shape[-1])).reshape(batch_size, n_timesteps, -1)

        # self.transformer
        tgt = torch.zeros((batch_size, n_timesteps + 1, self.output_dim)).to(src.device)
        for t in range(n_timesteps):
          output = self.transformer_decoder(tgt, reshaped_memory)
          tgt[:, t + 1] = output[:, t]

        tgt = tgt[:, 1:]
        return tgt

class ST_LR(nn.Module):
    '''
    Structured Inference Network (ST-LR) from Section 4 of the paper

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent variables
    obs_dim : int
        Dimension of the observations
    '''
    def __init__(self, latent_dim:int, obs_dim:int):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        
        self.transformer = CustomTransformer(obs_dim, latent_dim*2, 1, 1) # change 1 to something

        self.soft_plus = nn.Softplus()


    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        ST-LR

        Parameters
        ----------
        x : torch.Tensor
            Observations, shape = (batch_size, n_time_steps, obs_dim)

        Returns
        -------
        z_mus : torch.Tensor
            Mean for the latent variables, shape = (batch_size, n_time_steps, latent_dim)
        z_sigmas : torch.Tensor
            Standard deviation for the latent variables, shape = (batch_size, n_time_steps, latent_dim)
        z_samples : torch.Tensor
            Sampled latent variables, shape = (batch_size, n_time_steps, latent_dim)

        Raises
        ------
        ValueError
            If x does not have 3 dimensions (batch_size, n_time_steps, obs_dim)
            If obs_dim is not equal to self.obs_dim
        '''
        if len(x.shape) != 3:
            raise ValueError(f'Expected x to have 3 dimensions, got {len(x.shape)}')

        batch_size, n_time_steps, obs_dim = x.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f'Expected obs_dim to be {self.obs_dim}, got {obs_dim}')
  
        tgt = self.transformer(x)
        tgt = tgt.reshape(batch_size, n_time_steps, 2, self.latent_dim)

        # gather z_mus and z_sigmas from transformer output
        z_mus = tgt[:, :, 0, :]
        z_sigmas = tgt[:, :, 1, :]
        z_sigmas = self.soft_plus(z_sigmas)
        z_samples = z_mus + torch.normal(0, 1, z_sigmas.shape).to(z_mus.device)*z_sigmas

        return z_mus, z_sigmas, z_samples

class GatedTransitionFunction(nn.Module):
    def __init__(self, input_size, hidden_size = None):
        super().__init__()
        self.input_size = input_size
        if hidden_size is None:
            hidden_size = input_size
        self.hidden_size = hidden_size

        self.W_1g = nn.Linear(input_size, hidden_size)
        self.W_2g = nn.Linear(hidden_size, input_size)
        self.W_1h = nn.Linear(input_size, hidden_size)
        self.W_2h = nn.Linear(hidden_size, input_size)

        self.W_mp = nn.Linear(input_size, input_size)
        self.W_sp = nn.Linear(input_size, input_size)
        self.init_W_mp()

    def forward(self, z_t_1):
        # z_t_1: (batch_size, input_size)

        g_t = F.sigmoid(self.W_2g(F.relu(self.W_1g(z_t_1))))
        h_t = self.W_2h(F.relu(self.W_1h(z_t_1)))
        m_t = (1-g_t)*(self.W_mp(z_t_1)) + g_t*h_t
        s_t = F.softplus(self.W_sp(F.relu(h_t)))

        return m_t, s_t

    def init_W_mp(self):
        # Initialize W_mp to have identity weights and 0 bias
        self.W_mp.weight.data.copy_(torch.eye(self.input_size))
        self.W_mp.bias.data.fill_(0)


class EmissionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, observation_size):
        super(EmissionNetwork, self).__init__()

        # input layer size (size of z)
        self.input_size = input_size

        # observation x size
        self.observation_size = observation_size

        # hidden size defaults to input size if not given
        if hidden_size is None:
            self.hidden_size = self.input_size
        else:
            self.hidden_size = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.observation_size)
        )
        self.layers2 = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.observation_size),
            nn.Softplus()
        )

    def forward(self, x):
        return self.layers(x), self.layers2(x)
    

class VLBLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
            self,
            # Output of the ST-LR
            z_mus,
            z_sigmas,
            # Output of the emission network (for each z_sample)
            emission_mus,
            emission_sigmas,
            # Output of the transition network (again, for each z_sample)
            transition_mus,
            transition_sigmas,
            # Ground truth observations
            x_gt
        ):

        # This is the distribution of p(z0). Inserting this here simplifies control flow in the for loop.
        transition_mus    = torch.cat([torch.zeros_like(z_mus[:,0:1]), transition_mus], dim=1)
        transition_sigmas = torch.cat([torch.ones_like(z_sigmas[:,0:1]), transition_sigmas], dim=1)

        loss = 0
        # This for loop calculates the two summations in equation (6)
        T = z_mus.shape[1]

        for t in range(T):
            # This is p(xt | zt)
            observation_distribution = td.normal.Normal(emission_mus[:,t], emission_sigmas[:,t])

            # This is q(zt | z(t-1), x)
            # print(z_mus[:,t].shape)
            posterior_distribution = td.normal.Normal(z_mus[:,t], z_sigmas[:,t])

            # Add the expected value of ln p(xt | zt), with respect to q(zt | z(t-1), x).
            # We approximate the expected value with a single sample.
            loss += observation_distribution.log_prob(x_gt[:,t].reshape(-1,1))

            # Sample z(t-1) according to q(z(t-1) | z(t-2), x)
            # This then gives us two distributions for z(t):
            #   - The estimated posterior q(z(z) | z(t-1), x)
            #   - The transition distribution p(z(t) | z(t-1))

            loss -= td.kl_divergence(
                posterior_distribution,                                            # q(zt | z(t-1), x)
                td.normal.Normal(transition_mus[:,t], transition_sigmas[:,t])          # p(zt | z(t-1))
            )

        return -1*loss/T

    
