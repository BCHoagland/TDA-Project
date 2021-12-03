#############################################################################
# GIVES EXAMPLE OF MAKING POINTS IN LATENT SPACE OF AUTOENCODER SPREAD OUT  #
#############################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import AE, Rips
from data import MNIST


# fix random seed
torch.manual_seed(0)
np.random.seed(0)

# hyperparameters
num_epochs = 100
batch_size = 128
save_iter = 5
data_size = 28 * 28
n_h = 10
n_latent = 2

# models
model = AE(data_size, n_h, n_latent)
rips = Rips(max_edge_length=0.5)

# data
dataset = MNIST(batch_size)
data, labels = next(dataset.batches(labels=True))
val_data = [0] * 10
for i in range(10):
    idx = (labels == i).nonzero()
    val_data[i] = data[idx].squeeze()

# loss based on persistent homology death times
def h_loss(pts):
    deaths = rips(pts)
    total_0pers = torch.sum(deaths)
    # total_1pers = torch.sum(dgm1[:, 1] - dgm1[:, 0])
    # total_0pers = torch.sum(deaths ** 2)
    disk = (pts ** 2).sum(-1) - 1
    disk = torch.max(disk, torch.zeros_like(disk)).sum()
    # return -total_0pers -total_1pers + 1*disk
    return -total_0pers + 1*disk


# save original latent embeddings
with torch.no_grad():
    orig_pts = model.encode(val_data[0])

# train model to minimize homology-based loss
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
losses = []
for i in range(1000):
    pts = model.encode(val_data[0])
    opt.zero_grad()
    l = h_loss(pts)
    l.backward()
    losses.append(l.detach().item())
    opt.step()

# plotting
with torch.no_grad():
    # plot losses from training
    plt.plot(losses)
    plt.show()

    # plot original embedded points vs newly embedded points
    plt.clf()
    plt.scatter(orig_pts.numpy()[:, 0], orig_pts.numpy()[:, 1], label='original')

    pts = model.encode(val_data[0])
    plt.scatter(pts.numpy()[:, 0], pts.numpy()[:, 1], label='optimized')

    plt.legend()
    plt.show()