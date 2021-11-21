import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, latent_space_dim, lr):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(8,16,3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32,128),
            nn.ReLU(),
            nn.Linear(128,latent_space_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim,128),
            nn.ReLU(),
            nn.Linear(128,3*3*32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32,3,3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        # sigmoid is used to make all output pixels between 0 and 1
        return torch.sigmoid(self.decoder(x))

    def _optimize(self, val):
        self.optimizer.zero_grad()
        val.backward()
        self.optimizer.step()
    
    def minimize(self, val):
        self._optimize(val)
    
    def maximize(self, val):
        self._optimize(-val)
