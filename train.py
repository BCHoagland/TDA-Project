from torch.nn.functional import binary_cross_entropy

from project.hyperparams import Hyperparams
from project.trainer import Trainer


hyperparams = Hyperparams(
    seed = 0,
    num_epochs = 100,
    batch_size = 512,
    save_iter = 1,
    lr = 3e-4,
    data_size = 28 * 28,
    n_h = 64,
    n_latent = 2,
    top_coef = 5e-3
)
t = Trainer('AE_Model', 'AE', binary_cross_entropy, 'MNIST', hyperparams)
t.train()
