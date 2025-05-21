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
for dataset in "SelfDrivingCar" "ReID" "GeoDE" "FishDetection" "iWildCam" "AutoArborist"; do
    for embedding_type in "clip" "dino"; do
        for split in "train" "val1" "val2" "val3" "val4"; do
            # Running embedding script
            echo "Processing dataset: $dataset, embedding type: $embedding_type, split: $split"
            embedding_path="all_datasets/$dataset/embeddings/$embedding_type_$split.npy"
            python baselines/get_embeddings.py --dataset_name "$dataset" --embedding_type "$embedding_type" --split "$split" --save_path $embedding_path
            
            # Check if the embedding file was created successfully
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create embedding file for $dataset, $embedding_type, $split"
                continue
            fi
            
            # Run the centroid calculation script
            echo "Calculating centroids for dataset: $dataset, embedding type: $embedding_type, split: $split"
            python baselines/get_centroids.py --dataset_name "$dataset" --embedding_path embedding_path --split "$split" --save_folder "all_datasets/$dataset/embeddings/"

            # Check if the centroid file was created successfully
            if [ $? -ne 0 ]; then
                echo "Error: Failed to create centroid file for $dataset, $embedding_type, $split"
                continue
            fi
        done
    done
done



