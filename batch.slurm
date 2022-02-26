#!/bin/bash

#SBATCH --gres mps:100
#SBATCH --cpus-per-task 4
#SBATCH --mem 20

source .venv/bin/activate

/usr/bin/time -v python train_test.py