#!/bin/bash

#SBATCH --gres mps:100
#SBATCH --cpus-per-task 8
#SBATCH --mem 40
#SBATCH --job-name=SUIM

source .venv/bin/activate

/usr/bin/time -v python train.py