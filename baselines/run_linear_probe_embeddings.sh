#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --output=slurm/slurm-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --qos vision-beery-free


source /data/vision/beery/scratch/neha/.bashrc
conda activate datacomp

python saving_clip_embeddings_for_linear_probe.py \
	--dataset_name "$1" \
	--split "$2" \