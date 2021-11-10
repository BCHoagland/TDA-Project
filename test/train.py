import torch
from torch.nn.functional import binary_cross_entropy
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt

from model import VAE
from data import MNIST


# hyperparameters
num_epochs = 50
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

plot_latent(val_data, model, 0)

# start training
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataset.batches():
        out, mean, log_var = model(data)
        loss = binary_cross_entropy(out, data) + KL(mean, log_var)
        model.minimize(loss)

    # save images periodically
    if epoch % save_iter == save_iter - 1:
        img = out.data.view(out.size(0), 1, 28, 28)
        path = f'img/VAE/'
        Path(path).mkdir(parents=True, exist_ok=True)
        save_image(img, path + str(epoch + 1) + '_epochs.png')

        plot_latent(val_data, model, epoch + 1)

    # report loss
    print(f'Epoch {epoch}: loss {loss.item()}')

quit()

# generate new random images
input = torch.randn(96, 10)
out = vae.decode(input)
img = out.data.view(96, 1, 28, 28)
save_image(img, './generated_img/img.png')
