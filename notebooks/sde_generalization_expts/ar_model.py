import torch
from torch import nn


class ARModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1_mu = nn.Parameter(torch.zeros(1))
        self.layer1_sigma = nn.Parameter(torch.ones(1))

        self.layer2 = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.softplus = nn.Softplus()

    def log_prob(self, x):
        """
        Compute the log probability of the given samples

        Parameters
        ----------
        x : torch.Tensor
            shape = (batch_size, 2)
        """
        log_prob_1 = torch.distributions.Normal(
            self.layer1_mu, self.layer1_sigma
        ).log_prob(x[:, 0])

        dist2 = self.layer2(x[:, 0:1])
        mu, sigma = dist2[:, 0], self.softplus(dist2[:, 1])
        log_prob_2 = torch.distributions.Normal(mu, sigma).log_prob(x[:, 1])
        return log_prob_1 + log_prob_2

    def get_mu_sigma(self, x):
        mus = []
        sigmas = []
        mus.append(self.layer1_mu)
        sigmas.append(self.layer1_sigma)

        dist2 = self.layer2(x[:, 0:1])
        mu, sigma = dist2[:, 0], self.softplus(dist2[:, 1])
        mus.append(mu)
        sigmas.append(sigma)
        return mus, sigmas

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Sample from the model

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        torch.Tensor
            shape = (n_samples, 2)
        """
        x = torch.zeros((n_samples, 2))
        x[:, 0] = torch.distributions.Normal(
            torch.ones(x.shape[0]) * self.layer1_mu,
            torch.ones(x.shape[0]) * self.layer1_sigma,
        ).sample()
        dist2 = self.layer2(x[:, 0:1])
        mu, sigma = dist2[:, 0], self.softplus(dist2[:, 1])
        x[:, 1] = torch.distributions.Normal(mu, sigma).sample()
        return x


def train_ar_model(dataloader, device="cpu", verbose=False):
    model = ARModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1000):
        for x in dataloader:
            x = x.float().to(device)
            loss = -model.log_prob(x).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return model
