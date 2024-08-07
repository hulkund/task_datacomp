#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --qos vision-beery-free

source /data/vision/beery/scratch/neha/.bashrc
conda activate datacomp


python baselines/clip_linear_probe.py \
	--dataset_name "$1" \
	--subset_path "$2" \
    --outputs_path "$3"

