import random
import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np
import wandb

from model import AE, Rips
from data import MNIST


def plot_latent(val_data_by_class, model, epoch, topological, show=False):
    with torch.no_grad():
        plt.clf()
        for i in range(10):
            encoded = model.encode(val_data_by_class[i])
            plt.scatter(encoded[:,0], encoded[:,1])

        if show:
            plt.show()
        else:
            path = 'img/latent/AE-top/' if topological else 'img/latent/AE/'
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.savefig(path + str(epoch) + '.png')


# start training
def train(**config):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # fix random seed
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # get data
    dataset = MNIST(config['batch_size'])
    # val_data, labels = next(dataset.batches(labels=True))
    # val_data_by_class = [0] * 10
    # for i in range(10):
    #     idx = (labels == i).nonzero()
    #     val_data_by_class[i] = val_data[idx].squeeze()

    # models
    rips = Rips(max_edge_length=0.5).to(device)
    model = AE(config['data_size'], config['lr'], config['n_h'], config['n_latent']).to(device)

    # loss based on 0-dimensional persistent homology death times
    # small death time = bad
    # outside unit disk = bad
    # pts should already be on GPU
    def h_loss(pts):
        deaths = rips(pts)
        total_0pers = torch.sum(deaths)
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        return -total_0pers + 1*disk

    # training loop
    epoch_losses = []
    for epoch in range(config['num_epochs']):

        # minibatch optimization with Adam
        batch_losses = []
        for data in dataset.batches():
            data = data.to(device)

            enc = model.encode(data)
            out = model.decode(enc)
            # out = model(data)
            ae_loss = binary_cross_entropy(out, data)

            #! much better idea: enforce the topological loss on just each val_data[i] to encourage the regularization *per class*, not overall
            #! of course, should do some exploration first. See what happens in the latent space...

            if config['topological']:
                topological_loss = h_loss(enc)     #! might have to fix later if the latent space is made more than 2D?
            else:
                topological_loss = 0

            model.minimize(ae_loss + topological_loss)

            with torch.no_grad():
                batch_losses.append(ae_loss.cpu().item())

        with torch.no_grad():
            # save images periodically
            if epoch % config['save_iter'] == config['save_iter'] - 1:
                with torch.no_grad():
                    # example generated images
                    z = torch.randn(96, config['n_latent']).to(device)
                    img = model.decode(z).view(-1, 1, 28, 28).cpu()
                    path = 'img/AE-top/' if config['topological'] else 'img/AE/'
                    Path(path).mkdir(parents=True, exist_ok=True)
                    save_image(img, path + str(epoch + 1) + '_epochs.png')

                    # plot example points in latent space
                    # reduce dimension to 2D in order to plot
                    # plot_latent(val_data_by_class, model, epoch + 1, topological)

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
        topological = True,
        seed = 0,
        num_epochs = 100,
        batch_size = 128,
        save_iter = 20,
        lr = 3e-4,
        data_size = 28 * 28,
        n_h = 64,
        n_latent = 4
    )

wandb.init(project='TDA-autoencoders', entity='bchoagland', config=defaults)
config = wandb.config
train(**defaults)

# plt.clf()
# plt.plot(l1, label='standard')
# plt.plot(l2, label='h')
# plt.legend()
# plt.show()
# quit()
