#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=1-12:00:00
#SBATCH --chdir=/data/vision/beery/scratch/evelyn/task_datacomp
#SBATCH --requeue

source /data/vision/beery/scratch/evelyn/.bashrc

 
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