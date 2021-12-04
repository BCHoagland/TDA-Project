import random
import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
import wandb

from model import AE, VAE, Rips
from data import MNIST


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def KL(μ, log_var, batch_size, data_size):
    kl = -0.5 * torch.sum(1 + log_var - μ.pow(2) - log_var.exp())
    kl /= batch_size * data_size
    return kl


# Base loss for AE and VAE, and also the encoded data
def ae_loss_and_enc(model, data, config):
    if config['model'] == 'AE':
        enc = model.encode(data)
        out = model.decode(enc)
        return binary_cross_entropy(out, data), enc
    # vae_loss
    elif config['model'] == 'VAE':
        enc, μ, log_var = model.encode(data, return_extra=True)
        out = model.decode(enc)
        return binary_cross_entropy(out, data) + KL(μ, log_var, config['batch_size'], config['data_size']), enc


# loss based on 0-dimensional persistent homology death times
# small death time = bad
# outside unit disk = bad
# pts should already be on GPU
def h_loss(pts, rips):
    deaths = rips(pts)
    total_0pers = torch.sum(deaths)
    disk = (pts ** 2).sum(-1) - 1
    disk = torch.max(disk, torch.zeros_like(disk)).sum()
    return -total_0pers + 1*disk


# take data and labels and split the data up based on class
# returns a list; index i in the list = data points for class i
# since we're using MNIST, this list is length 10
def get_data_by_class(data, labels):
    data_by_class = [0] * 10
    for i in range(10):
        idx = (labels == i).nonzero()
        data_by_class[i] = data[idx].squeeze().to(device)
    return data_by_class


# log sample generated images
def log_generated_images(model, config, epoch):
    z = torch.randn(96, config['n_latent']).to(device)
    img = model.decode(z).view(-1, 1, 28, 28).cpu()
    wandb.log({f'img': wandb.Image(img)}, step=epoch + 1)


# plot points (divided by class) in latent space
#! reduce dimension to 2D in order to plot
def log_latent_embeddings(model, val_data_by_class, epoch):
    plt.clf()
    for i in range(10):
        encoded = model.encode(val_data_by_class[i])
        plt.scatter(encoded[:,0], encoded[:,1])
    wandb.log({'latent': wandb.Image(plt)}, step=epoch + 1)


# start training
def train(**config):
    # fix random seed
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # get data
    dataset = MNIST(config['batch_size'])
    val_data_by_class = get_data_by_class(*next(dataset.batches(labels=True)))

    # models
    rips = Rips(max_edge_length=0.5).to(device)
    if config['model'] == 'AE':
        model_type = AE
    elif config['model'] == 'VAE':
        model_type = VAE
    model = model_type(config['data_size'], config['lr'], config['n_h'], config['n_latent']).to(device)

    # training loop
    epoch_losses = []
    for epoch in range(config['num_epochs']):

        # minibatch optimization with Adam
        batch_losses = []
        for data in dataset.batches():
            data = data.to(device)

            #! much better idea: enforce the topological loss on just each val_data[i] to encourage the regularization *per class*, not overall
            #! of course, should do some exploration first. See what happens in the latent space...

            ae_loss, enc = ae_loss_and_enc(model, data, config)
            topological_loss = h_loss(enc, rips) if config['topological'] else 0
            print(config['topological'], config['model'])
            quit()

            model.minimize(ae_loss + topological_loss)

            with torch.no_grad():
                batch_losses.append(ae_loss.cpu().item())

        with torch.no_grad():
            # save images periodically
            if epoch % config['save_iter'] == config['save_iter'] - 1:
                # log images and latent embeddings
                log_generated_images(model, config, epoch)
                if config['n_latent'] == 2:
                    log_latent_embeddings(model, val_data_by_class, epoch)

            # report loss
            avg_epoch_loss = sum(batch_losses) / len(batch_losses)
            wandb.log({'Average epoch loss': avg_epoch_loss}, step=epoch + 1)
            # print(f'Epoch {epoch + 1}: loss {avg_epoch_loss}')
            epoch_losses.append(avg_epoch_loss)

    return epoch_losses


############
# TRAINING #
############
defaults = dict(
        model = 'VAE',
        topological = True,
        seed = 0,
        num_epochs = 10,
        batch_size = 128,
        save_iter = 1,
        lr = 3e-4,
        data_size = 28 * 28,
        n_h = 64,
        n_latent = 2
    )

wandb.init(project='TDA-autoencoders', entity='bchoagland', config=defaults)
config = wandb.config
train(**config)
