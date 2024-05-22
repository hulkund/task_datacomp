#!/bin/bash

datasets=("iWildCam" "COOS" "FMoW")
splits=("train" "test1" "test2" "test3" "test4")

# Loop through a range of 5 numbers (0 to 4)
for dataset in "${datasets[@]}"; do
  # Nested loop through another range of 5 numbers (0 to 4)
  for split in "${splits[@]}"; do
    sbatch run_linear_probe_embeddings.sh "$dataset" "$split"
  done
done

