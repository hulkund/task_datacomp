#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/queryset-%J.out
#SBATCH -c 8
#SBATCH -w beery-a100-1
#SBATCH --mem=50G
#SBATCH --time=12:00:00

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.local_paths.sh" ]; then
    source "$SCRIPT_DIR/.local_paths.sh"
fi

BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"
if [ -f "$BASHRC_PATH" ]; then
    source "$BASHRC_PATH"
fi

learning_rates=(0.001 0.01 0.1)
batch_sizes=(32)
dataset_names=(GeoDE iWildCam AutoArborist)

for dataset_name in "${dataset_names[@]}"
do 
    for lr in "${learning_rates[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do
            python baselines/train_on_query.py --outputs_path experiments/$dataset_name/queryset --dataset_name $dataset_name --dataset_config configs/datasets.yaml --batch_size $batch_size --lr $lr 
        done
    done
done

# for lr in "${learning_rates[@]}"
# do
#     for batch_size in "${batch_sizes[@]}"
#     do
#         python baselines/train_on_subset.py --outputs_path experiments/GeoDE/oracle_set/ --dataset_name GeoDE --dataset_config configs/datasets.yaml --batch_size $batch_size --lr $lr --subsets_path
#     done
# done
