# StyleGAN-TDA
Persistent homology and generative models

## what's in here
All the relevant code is in the `project` folder. `project/tests` contains individual test scripts that were used to explore various ideas. `train.py` is the main training script. The various models that are used are defined in `model.py`, and the dataloader is defined in `data.py`


# Dependencies
Just create a virtual environment using the Pipfile. But right now the dependencies are:
* torch
* torchvision
* numpy
* matplotlib
* gudhi
* sklearn
* wandb