#!/bin/bash

#SBATCH --gres gpu:RX6900XT:1
#SBATCH --cpus-per-task 4
#SBATCH --mem 20

source .venv.rocm/bin/activate

/usr/bin/time -v python train_test.py