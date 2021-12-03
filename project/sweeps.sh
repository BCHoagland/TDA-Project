#? GPU support

#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

pip3 install --user pipenv

export WANDB_API_KEY="cfae9d07743626b29d0f1e05a85acb9ec627808d"

git clone https://github.com/BCHoagland/TDA-Project.git
cd TDA-Project
pipenv shell
cd project
wandb sweep #? SWEEP_ID
