import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt
import gudhi as gd

from model import VAE, RipsH1
from data import MNIST


# hyperparameters
num_epochs = 2
batch_size = 128
save_iter = 1

data_size = 28 * 28
n_h = 128
n_latent = 2


def KL(mean, log_var):
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    kl /= batch_size * 28 * 28
    return kl


# instantiate model and dataset
model = VAE(data_size, n_h, n_latent)
dataset = MNIST(batch_size)

# fix data that we wil track throughout training
data, labels = next(dataset.batches(labels=True))
val_data = [0] * 10
for i in range(10):
    idx = (labels == i).nonzero()
    val_data[i] = data[idx].squeeze()


def plot_latent(val_data, model, epoch):
    with torch.no_grad():
        for i in range(10):
            encoded = model.encode(val_data[i])
            plt.scatter(encoded[:,0], encoded[:,1])

        path = f'img/latent/VAE/'
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
    rips = RipsH1(max_edge_length=0.5)

    def h1_loss(pts):
        dgm1 = rips(pts)
        total_1pers = torch.sum(dgm1[:, 1] - dgm1[:, 0])
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        return -total_1pers + 1*disk

    with torch.no_grad():
        orig_pts = model.encode(val_data[0])
        # plt.clf()
        # plt.scatter(orig_pts.numpy()[:, 0], orig_pts.numpy()[:, 1])
        # plt.show()

    from torch.optim.lr_scheduler import LambdaLR
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    # scheduler = LambdaLR(opt, [lambda epoch: 10. / (10 + epoch)])
    losses = []

    for i in range(1000):
        pts = model.encode(val_data[0])
        opt.zero_grad()
        l = h1_loss(pts)
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

# ripts_test()
# ==========

# start training
def train(topological):
    rips = RipsH1(max_edge_length=0.5)

    def h1_loss(pts):
        dgm1 = rips(pts)
        total_1pers = torch.sum(dgm1[:, 1] - dgm1[:, 0])
        disk = (pts ** 2).sum(-1) - 1
        disk = torch.max(disk, torch.zeros_like(disk)).sum()
        return -total_1pers + 1*disk

    model = VAE(data_size, n_h, n_latent)

    epoch_losses = []
    for epoch in range(num_epochs):

        # minibatch optimization with Adam
        batch_losses = []
        for data in dataset.batches():
            out, mean, log_var = model(data)
            vae_loss = binary_cross_entropy(out, data) + KL(mean, log_var)

            if topological:
                topological_loss = h1_loss(out)     #! might have to fix later if the latent space is made more than 2D?
            else:
                topological_loss = 0

            model.minimize(vae_loss + topological_loss)

            with torch.no_grad():
                batch_losses.append(vae_loss.item())

        # save images periodically
        if epoch % save_iter == save_iter - 1:
            img = out.data.view(out.size(0), 1, 28, 28)
            path = f'img/VAE/'
            Path(path).mkdir(parents=True, exist_ok=True)
            save_image(img, path + str(epoch + 1) + '_epochs.png')

            plot_latent(val_data, model, epoch + 1)

        # report loss
        print(f'Epoch {epoch}: loss {vae_loss.item()}')
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return epoch_losses

l1 = train(topological=False)
l2 = train(topological=True)
plt.clf()
plt.plot(l1, label='standard')
plt.plot(l2, label='h1')
plt.show()
quit()

# generate new random images
input = torch.randn(96, 10)
out = vae.decode(input)
img = out.data.view(96, 1, 28, 28)
save_image(img, './generated_img/img.png')
