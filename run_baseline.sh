#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/filter-%J.out
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=1-12:00:00
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

conda init
conda activate "$CONDA_ENV"
 
# Echo the positional parameters so they're visible in the job output/logs
echo "Running run_baseline.sh with parameters:  name: $1  embedding_path: $2  save_path: $3  fraction: $4  val_embedding_path: $5  centroids_path: $6  extra args: ${@:7}"

python baselines.py \
    --name "$1" \
	--embedding_path "$2" \
	--save_path "$3" \
	--fraction $4 \
    --val_embedding_path "$5" \
    --centroids_path "$6" \
    "${@:7}"
