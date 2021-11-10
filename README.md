# StyleGAN-TDA
Persistent homology and generative models

## what's in here so far
Go to the `test` folder and run `python train.py` to train a variational autoencoder on MNIST. It'll plot out the latent representations of a subset of the training data every epoch, as well as giving example generated images.

# Dependencies
Just create a virtual environment using the Pipfile. But right now the dependencies are:
* torch
* torchvision
* numpy
* matplotlib
* gudhi
* sklearn