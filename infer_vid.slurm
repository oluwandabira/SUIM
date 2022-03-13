#!/bin/bash

#SBATCH --gres mps:100
#SBATCH --cpus-per-task 8
#SBATCH --mem 40
#SBATCH --job-name "Infer Vid"

source .venv/bin/activate

/usr/bin/time -v python infer_vid.py $@