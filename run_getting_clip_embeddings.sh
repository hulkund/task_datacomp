#!/bin/bash
#SBATCH --partition=vision-shared-l40s
#SBATCH --qos=lab-free
#SBATCH --account=vision-beery
#SBATCH --output=slurm/clip-embeddings-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/data/vision/beery/scratch/evelyn/task_datacomp
#SBATCH --requeue

source /data/vision/beery/scratch/evelyn/.bashrc
conda activate datacomp

# for dataset in "GeoDE" "iWildCam" "AutoArborist"; do
for dataset in "iWildCam"; do
    for embedding_type in "clip"; do
        for split in "val1" "val2" "val3" "val4" "test1" "test2" "test3" "test4" "train"; do
            # Running embedding script
            echo "Processing dataset: $dataset, embedding type: $embedding_type, split: $split"
            embedding_path="all_datasets/${dataset}/embeddings/${embedding_type}_${split}.npy"
            python baselines/get_embeddings.py --dataset_name "$dataset" --embedding_type "$embedding_type" --split "$split" --save_path $embedding_path
            
            # Check if the embedding file was created successfully
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create embedding file for $dataset, $embedding_type, $split"
                continue
            fi
        done
    done
done

