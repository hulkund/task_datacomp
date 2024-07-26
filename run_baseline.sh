#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --time=12:00:00

source /data/vision/beery/scratch/neha/.bashrc
conda activate datacomp


python baselines.py \
    --name "$1" \
	--embedding_path "$2" \
	--save_path "$3" \
	--fraction $4 \
    --val_embedding_path "$5" \
    --centroids_path "$6" \

