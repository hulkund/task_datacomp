#!/bin/bash
#SBATCH --partition=vision-beery-l40s
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --output=slurm/slurm-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=6-00:00:00
#SBATCH --chdir=/data/vision/beery/scratch/evelyn/task_datacomp
#SBATCH --requeue

python3 tune_baselines/run_baseline_manager.py
