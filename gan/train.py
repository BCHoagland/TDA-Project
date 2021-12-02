import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import random
import os
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.optim as optim

import gudhi as gd
import gudhi.representations

from model import GAN

torch.manual_seed(0)


#----------------------------------------
# Datasets
#----------------------------------------
data_dir = '../data'
train_dataset = torchvision.datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=False, download=True)

m = len(train_dataset)
batch_size = 256
latent_space_dim = 3
lr = 1e-2
num_epochs = 20

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

#----------------------------------------
# Make sure the model works
#----------------------------------------
model = GAN(batch_size=batch_size, latent_space_dim=latent_space_dim, lr=lr)

print('-' * 50)

img, _ = test_dataset[0]
img = img.unsqueeze(0) # Add the batch dimension in the first axis
print('Original image shape:', img.shape)

img_disc = model.discriminate(img)
print('Discriminator output shape (training data):', img_disc.shape)

img_gen = model.generate(1)
print('Decoded image shape:', img_gen.shape)

img_disc = model.discriminate(img_gen)
print('Discriminator output shape (generated data):', img_disc.shape)

print('-' * 50)

#----------------------------------------
# Training/testing methods
#----------------------------------------

# training epoch: get data from training dataloader and optimize model
def train_epoch(model):
    for img, _ in train_loader:

        # check that batch is full; if not, will run into dimensionality errors
        if img.shape[0] == batch_size:
            model.optimize_D(img)
            model.optimize_G()

# save sample generated images from model
def gen_images(model, id):
    img = model.generate(96).view(96, 1, 28, 28)

    path = f'img/'
    Path(path).mkdir(parents=True, exist_ok=True)
    save_image(img, path + str(id + 1) + '.png')

#----------------------------------------
# Training loop
#----------------------------------------
for epoch in tqdm(range(num_epochs)):
    train_epoch(model)
    gen_images(model, epoch)
