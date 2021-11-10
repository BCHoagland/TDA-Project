import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, data_size, n_h, n_latent) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(data_size, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU()
        )

        self.mean = nn.Sequential(
            nn.Linear(n_h, n_latent)
        )

        self.log_var = nn.Sequential(
            nn.Linear(n_h, n_latent)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, data_size),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.encoder(x)

        mean = self.mean(x)
        log_var = self.log_var(x)

        # z = mean + (std_dev * eps), where eps ~ N(0,1)
        z = mean + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))

        x_hat = self.decoder(z)

        return x_hat, mean, log_var
    
    def encode(self, x):
        x = self.encoder(x)

        mean = self.mean(x)
        log_var = self.log_var(x)
        z = mean + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        return z

    def decode(self, z):
        return self.decoder(z)
    
    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def minimize(self, loss):
        self._optimize(loss)
    
    def maximize(self, loss):
        self._optimize(-loss)
