import sys
import random
import torch
import numpy as np

from data import Dataset
from models import *
from algos import *


#! fix 'model' and 'algo' so you don't have to import all
#! send in E and D architectures, not models themselves


class Hyperparams(dict):

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)


class Trainer():
    def __init__(self, model, algo, dataset, hyperparams, log_online) -> None:
        self.model = getattr(sys.modules[__name__], model)
        self.algo = getattr(sys.modules[__name__], algo)
        self.dataset = Dataset(dataset, hyperparams.batch_size)
        self.params = hyperparams
        self.log_online = log_online

    def train(self):
        # fix random seed
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

        # get data
        dataset = Dataset(self.params.dataset, self.params.batch_size)

        # training loop
        for epoch in range(self.params.num_epochs):

            # minibatch optimization
            for batch in dataset.batches():
                loss = algo.loss(batch)




hyperparams = Hyperparams(
    batch_size = 128
)
Trainer('AE_Model', 'AE', 'MNIST', hyperparams, None)
