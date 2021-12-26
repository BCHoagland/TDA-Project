import torch
import torch.nn as nn


class VAE_Model(nn.Module):
    def __init__(self, data_size, lr, n_h, n_latent) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(data_size, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU()
        )

        self.μ = nn.Linear(n_h, n_latent)
        self.log_var = nn.Linear(n_h, n_latent)

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, data_size),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        quit('YOU SHOULDNT DO THAT')
    
    def encode(self, x):
        x = self.encoder(x)
        μ = self.μ(x)
        log_var = self.log_var(x)

        out = μ + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))

        if return_extra:
            return out, μ, log_var
        return out

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
