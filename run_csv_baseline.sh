#!/bin/bash
#SBATCH --partition=csail-shared
#SBATCH --qos=lab-free
#SBATCH --account=vision-beery
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=10G
#SBATCH --time=1:00:00

source /data/vision/beery/scratch/neha/.bashrc
conda activate datacomp


python baselines.py \
    --name "$1" \
	--dataset_name "$2" \
	--task_name "$3" \
    --fraction $4 \
	--save_path "$5" 

