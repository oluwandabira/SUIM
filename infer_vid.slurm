#!/bin/bash

#SBATCH --gres mps:40
#SBATCH --cpus-per-task 4
#SBATCH --mem 4G
#SBATCH --job-name "InferVid"

source .venv/bin/activate

/usr/bin/time -v python infer_vid.py $@