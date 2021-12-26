import sys
import random
import torch
import numpy as np
from tqdm import tqdm

from project.data import Dataset
from project.models import *
from project.algos import *


#! fix 'model' and 'algo' so you don't have to import all
#! send in E and D architectures, not models themselves
#! make 'data_size' dynamic


class Trainer():
    def __init__(self, model, algo, rec_loss_fn, dataset, hyperparams) -> None:
        self.model = getattr(sys.modules[__name__], model)(
            hyperparams.data_size,
            hyperparams.lr,
            hyperparams.n_h,
            hyperparams.n_latent
        )
        self.algo = getattr(sys.modules[__name__], algo)()
        self.rec_loss = rec_loss_fn
        self.dataset = Dataset(dataset, hyperparams.batch_size)
        self.params = hyperparams

    def train(self, ):
        # fix random seed
        random.seed(self.params.seed)
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)

        # training loop
        for epoch in tqdm(range(self.params.num_epochs)):

            # minibatch optimization
            batch_losses = []
            for batch in self.dataset.batches():
                loss = self.algo.loss(batch, self.model, self.rec_loss, self.params)
                self.model.minimize(loss)

                with torch.no_grad():
                    batch_losses.append(loss.item())
            
            # reporting
            with torch.no_grad():

                #! implement callbacks here
                pass
