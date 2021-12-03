import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np

from model import AE, Rips
from data import MNIST


# hyperparameters
num_epochs = 50
batch_size = 256
save_iter = 5

data_size = 28 * 28
n_h = 10
n_latent = 4                #! try this with 3 dimensions to see if you can plot it still

# fix data
#! redo how getting data works since this is messy
def get_data():
    dataset = MNIST(batch_size)
    val_data, labels = next(dataset.batches(labels=True))
    val_data_by_class = [0] * 10
    for i in range(10):
        idx = (labels == i).nonzero()
        val_data_by_class[i] = val_data[idx].squeeze()
    return dataset, val_data_by_class


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
def train(topological):
    # fix random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # get data
    dataset, val_data_by_class = get_data()

    # models
    rips = Rips(max_edge_length=0.5)
    model = AE(data_size, n_h, n_latent)

    # loss based on 0-dimensional persistent homology death times
    # small death time = bad
    # outside unit disk = bad
    def h_loss(pts):
        deaths = rips(pts)
        total_0pers = torch.sum(deaths)
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        return -total_0pers + 1*disk

    # training loop
    epoch_losses = []
    for epoch in range(num_epochs):

        # minibatch optimization with Adam
        batch_losses = []
        for data in dataset.batches():
            enc = model.encode(data)
            out = model.decode(enc)
            # out = model(data)
            ae_loss = binary_cross_entropy(out, data)

            #! much better idea: enforce the topological loss on just each val_data[i] to encourage the regularization *per class*, not overall
            #! of course, should do some exploration first. See what happens in the latent space...

            if topological:
                topological_loss = h_loss(enc)     #! might have to fix later if the latent space is made more than 2D?
            else:
                topological_loss = 0

            model.minimize(ae_loss + topological_loss)

            with torch.no_grad():
                batch_losses.append(ae_loss.item())

        # save images periodically
        if epoch % save_iter == save_iter - 1:
            with torch.no_grad():
                # example generated images
                z = torch.randn(96, n_latent)
                img = model.decode(z).view(-1, 1, 28, 28)
                path = 'img/AE-top/' if topological else 'img/AE/'
                Path(path).mkdir(parents=True, exist_ok=True)
                save_image(img, path + str(epoch + 1) + '_epochs.png')

                # plot example points in latent space
                # reduce dimension to 2D in order to plot
                # plot_latent(val_data_by_class, model, epoch + 1, topological)

        # report loss
        print(f'Epoch {epoch}: loss {ae_loss.item()}')
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return epoch_losses

#! figure out why training both makes topological training have weird latent space
l1 = train(topological=False)
l2 = train(topological=True)
# quit('done')
plt.clf()
plt.plot(l1, label='standard')
plt.plot(l2, label='h')
plt.legend()
plt.show()
quit()

# generate new random images
input = torch.randn(96, 10)
out = ae.decode(input)
img = out.data.view(96, 1, 28, 28)
save_image(img, './generated_img/img.png')
