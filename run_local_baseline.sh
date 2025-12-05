#!/bin/bash

# Echo the positional parameters so they're visible in the job output/logs
echo "Running run_baseline.sh with parameters:  name: $1  embedding_path: $2  save_path: $3  fraction: $4  val_embedding_path: $5  centroids_path: $6  supervsed: $7  random seed: $8  extra args: ${@:8}"

python baselines.py \
    --name "$1" \
	--embedding_path "$2" \
	--save_path "$3" \
	--fraction $4 \
    --val_embedding_path "$5" \
    --centroids_path "$6" \
    --supervised "$7" \
    --random_seed "$8" \
    "${@:9}"