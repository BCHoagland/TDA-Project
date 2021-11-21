import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import random
import os
from tqdm import tqdm 
import plotly.io as pio
# pio.renderers.default = 'colab'

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px

import gudhi as gd
import gudhi.representations

from model import Autoencoder


torch.manual_seed(0)


#----------------------------------------
# Datasets
#----------------------------------------
data_dir = 'data'

train_dataset = torchvision.datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=False, download=True)

m = len(train_dataset)
batch_size = 256
latent_space_dim = 3
lr = 1e-3
num_epochs = 3

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

#----------------------------------------
# Make sure the model works
#----------------------------------------
model = Autoencoder(latent_space_dim=latent_space_dim, lr=lr)

print('-' * 50)

img, _ = test_dataset[0]
img = img.unsqueeze(0) # Add the batch dimension in the first axis
print('Original image shape:', img.shape)

img_enc = model.encode(img)
print('Encoded image shape:', img_enc.shape)

dec_img = model.decode(img_enc)
print('Decoded image shape:', dec_img.shape)

print('-' * 50)

#----------------------------------------
# Training/testing methods
#----------------------------------------
def _loss(orig, decoded):
    mse_loss = ((decoded - orig) ** 2).mean()

    #! implement landscapes stuff
    topo_loss = 0

    #! coefficients
    return mse_loss + topo_loss

def _epoch(model, dataloader, optimize):
    losses = []

    for img, _ in dataloader:
        enc = model.encode(img)
        dec = model.decode(enc)

        loss = _loss(img, dec)
        with torch.no_grad():
            losses.append(loss.item())

        if optimize:
            model.minimize(loss)
    
    return np.mean(losses)

def train_epoch(model):
    return _epoch(model, train_loader, True)

def test_epoch(model):
    return _epoch(model, test_loader, False)

#----------------------------------------
# Training loop
#----------------------------------------
def train(model, epochs):
    print('Starting training')
    train_losses = []
    test_losses = []

    for _ in tqdm(range(epochs)):
        train_losses.append(train_epoch(model))
        test_losses.append(test_epoch(model))

    print('-' * 50)
    return train_losses, test_losses

train_losses, test_losses = train(model, num_epochs)
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.show()
