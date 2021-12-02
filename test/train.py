import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
import gudhi as gd
import numpy as np

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

# instantiate model and dataset
model = AE(data_size, n_h, n_latent)
dataset = MNIST(batch_size)

# fix data that we wil track throughout training
data, labels = next(dataset.batches(labels=True))
val_data = [0] * 10
for i in range(10):
    idx = (labels == i).nonzero()
    val_data[i] = data[idx].squeeze()


def plot_latent(val_data, model, epoch, topological):
    with torch.no_grad():
        for i in range(10):
            encoded = model.encode(val_data[i])
            plt.scatter(encoded[:,0], encoded[:,1])

        path = 'img/latent/AE-top/' if topological else 'img/latent/AE/'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + str(epoch) + '.png')
        # plt.show()

    # rips = gd.RipsComplex(
    #     distance_matrix = encoded,
    #     max_edge_length = 2
    # ).create_simplex_tree(max_dimension=2)
    # barcodes = rips.persistence()
    # rips.compute_persistence()
    # print(rips.betti_numbers())

# plot_latent(val_data, model, 0)


# ==========
def rips_test():
    rips = Rips(max_edge_length=0.5)

    def h_loss(pts):
        deaths = rips(pts)
        total_0pers = torch.sum(deaths)
        # total_1pers = torch.sum(dgm1[:, 1] - dgm1[:, 0])
        # total_0pers = torch.sum(deaths ** 2)
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        # return -total_0pers -total_1pers + 1*disk
        return -total_0pers + 1*disk

    # def h_loss(pts):
    #     dgm1 = rips(pts)
    #     total_1pers = torch.sum(dgm1[:, 1] - dgm1[:, 0])
    #     disk = (pts ** 2).sum(-1) - 1
    #     disk = torch.max(disk, torch.zeros_like(disk)).sum()
    #     return -total_1pers + 1*disk

    with torch.no_grad():
        orig_pts = model.encode(val_data[0])
        # plt.clf()
        # plt.scatter(orig_pts.numpy()[:, 0], orig_pts.numpy()[:, 1])
        # plt.show()

    # from torch.optim.lr_scheduler import LambdaLR
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    # scheduler = LambdaLR(opt, [lambda epoch: 10. / (10 + epoch)])
    losses = []

    for i in range(1000):
        pts = model.encode(val_data[0])
        opt.zero_grad()
        l = h_loss(pts)
        l.backward()
        losses.append(l.detach().item())
        opt.step()
        # scheduler.step()

    plt.clf()
    plt.plot(losses)
    plt.show()
    with torch.no_grad():
        plt.clf()
        plt.scatter(orig_pts.numpy()[:, 0], orig_pts.numpy()[:, 1], label='original')
        # plt.show()

        pts = model.encode(val_data[0])
        # plt.clf()
        plt.scatter(pts.numpy()[:, 0], pts.numpy()[:, 1], label='optimized')
        plt.legend()
        plt.show()

# rips_test()
# ==========

# start training
def train(topological):
    rips = Rips(max_edge_length=0.5)

    def h_loss(pts):
        deaths = rips(pts)
        total_0pers = torch.sum(deaths)
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        return -total_0pers + 1*disk

    model = AE(data_size, n_h, n_latent)

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

                plot_latent(val_data, model, epoch + 1, topological)

        # report loss
        print(f'Epoch {epoch}: loss {ae_loss.item()}')
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return epoch_losses

#! figure out why training both makes topological training have weird latent space
l1 = train(topological=False)
quit('stopped before doing topological training')
l2 = train(topological=True)
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
