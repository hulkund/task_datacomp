#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --output=slurm/clip_embeddings-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --time=2-20:00:00

source /data/vision/beery/scratch/neha/.bashrc
micromamba activate datacomp

# python get_clip_embeddings.py --dataset_name "SelfDrivingCar"
python get_clip_embeddings.py --dataset_name "ReID"
