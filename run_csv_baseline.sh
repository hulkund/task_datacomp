#!/bin/bash
#SBATCH --partition=vision-shared-h100,vision-shared-l40s
#SBATCH --qos=lab-free
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=23:00:00
#SBATCH --chdir=/data/vision/beery/scratch/evelyn/task_datacomp
#SBATCH --requeue

source /data/vision/beery/scratch/evelyn/.bashrc
conda activate datacomp


python baselines.py \
    --name "$1" \
	--dataset_name "$2" \
	--task_name "$3" \
    --fraction $4 \
	--save_path "$5" 

