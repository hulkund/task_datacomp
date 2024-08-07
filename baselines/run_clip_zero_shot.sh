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


# python clip_zero_shot.py --dataset_name COOS
python clip_zero_shot.py --dataset_name FMoW
# python clip_zero_shot.py --dataset_name iWildCam

