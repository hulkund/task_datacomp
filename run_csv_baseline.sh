#!/bin/bash
#SBATCH --partition=vision-shared-h100,vision-shared-l40s
#SBATCH --qos=lab-free
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=23:00:00
#SBATCH --chdir=.
#SBATCH --requeue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.local_paths.sh" ]; then
    source "$SCRIPT_DIR/.local_paths.sh"
fi

BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"
CONDA_ENV="${CONDA_ENV:-datacomp}"
if [ -f "$BASHRC_PATH" ]; then
    source "$BASHRC_PATH"
fi
conda activate "$CONDA_ENV"


python baselines.py \
    --name "$1" \
	--dataset_name "$2" \
	--task_name "$3" \
    --fraction $4 \
	--save_path "$5" 
