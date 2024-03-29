import torch
import torch.nn as nn
import gudhi as gd


class AE(nn.Module):
    def __init__(self, data_size, lr, n_h, n_latent) -> None:
        super().__init__()

        # self.encoder = nn.Sequential(
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
        #     nn.Linear(128,n_latent)
        # )
        
        # self.decoder = nn.Sequential(
        #     nn.Linear(n_latent,128),
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
        #     nn.Sigmoid()
        # )

        self.encoder = nn.Sequential(
            nn.Linear(data_size, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h),
            nn.ELU(),
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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)

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


class VAE(nn.Module):
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
    
    def encode(self, x, return_extra=False):
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


class Rips(nn.Module):
    def __init__(self, max_edge_length):
        super(Rips, self).__init__()
        self.max_edge_length = max_edge_length

    def forward(self, input):
        rips = gd.RipsComplex(points=input, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=0)
        st.compute_persistence()
        idx = st.flag_persistence_generators()

        # 1-D
        # if len(idx[1]) == 0:
        #     verts = torch.empty((0, 4), dtype=int)
        # else:
        #     verts = torch.tensor(idx[1][0])
        # dgm = torch.norm(input[verts[:, (0, 2)]] - input[verts[:, (1, 3)]], dim=-1)

        # 0-D
        if len(idx[0]) == 0:
            verts = torch.empty((0, 2), dtype=int)
        else:
            verts = torch.tensor(idx[0][:, 1:])
        dgm = torch.norm(input[verts[:,0], :] - input[verts[:,1], :], dim=-1)

        return dgm
