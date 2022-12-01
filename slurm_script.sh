#!/bin/bash

#SBATCH --job-name=covr-diversity
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --mem=150GB
#SBATCH --time=0
#SBATCH --output=outputs/slurm.%j.out
#SBATCH --error=outputs/slurm.%j.err

srun python -m experiments.parallel_driver exp-1 final0 --dataset "covr" --split-type "iid;template" --subsample-type "template;subtree" --n-gpus 8 --n-jobs-per-gpu 1 --train
