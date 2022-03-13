#!/bin/env bash

#SBATCH --gres mps:40
#SBATCH --cpus-per-task 4
#SBATCH --mem 4G
#SBATCH --job-name "InferVid"

readarray -t files < $1

video=${files[$SLURM_ARRAY_TASK_ID]}
outfolder=$(basename "$video" .AVI)

mkdir output

source .venv/bin/activate

/usr/bin/time -v python infer_vid.py "$video" "output/$outfolder"
