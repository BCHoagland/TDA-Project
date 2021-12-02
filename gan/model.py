import torch
from torch import nn


class GAN(nn.Module):
    def __init__(self, batch_size, latent_space_dim, lr):
        super().__init__()

        self.batch_size = batch_size
        self.latent_space_dim = latent_space_dim

        n_hidden = 128

        self.generator = nn.Sequential(
            nn.Linear(latent_space_dim, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 28 * 28),
            nn.Sigmoid()
        )

        self.discriminator = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
        
        # self.discriminator = nn.Sequential(
        #     nn.Conv2d(1,8,3,stride=2,padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(8,16,3,stride=2,padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16,32,3,stride=2,padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(3*3*32,128),
        #     nn.ReLU(),
        #     nn.Linear(128,latent_space_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_space_dim, 1),
        #     nn.Sigmoid()
        # )
        
        # self.generator = nn.Sequential(
        #     nn.Linear(latent_space_dim,128),
        #     nn.ReLU(),
        #     nn.Linear(128,3*3*32),
        #     nn.ReLU(),
        #     nn.Unflatten(dim=1, unflattened_size=(32,3,3)),
        #     nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        #     nn.Sigmoid() # all output pixels should be in [0, 1]
        # )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    # def forward(self, x):
    #     return self.decode(self.encode(x))
    
    def noise(self, batch_size=None):
        m = self.batch_size if batch_size is None else batch_size
        return torch.randn(m, self.latent_space_dim)
    
    def discriminate(self, x):
        return self.discriminator(x)
    
    def generate(self, batch_size=None):
        return self.generator(self.noise(batch_size))

    def _optimize(self, val):
        self.optimizer.zero_grad()
        val.backward()
        self.optimizer.step()
    
    def _minimize(self, val):
        self._optimize(val)
    
    def _maximize(self, val):
        self._optimize(-val)

    def optimize_D(self, x):
        G, D = self.generator, self.discriminator
        z = self.noise()
        d_loss = (torch.log(D(x)) + torch.log(1 - D(G(z)))).mean()
        self._maximize(d_loss)

    def optimize_G(self):
        G, D = self.generator, self.discriminator
        z = self.noise()
        # g_loss = torch.log(D(G(z))).mean()                                           # w/ trick            # TODO: decide whether or not to keep
        # G.maximize(g_loss)
        g_loss = torch.log(1 - D(G(z))).mean()                                       # w/out trick
        self._minimize(g_loss)
