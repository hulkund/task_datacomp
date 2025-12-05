#!/bin/bash

# Assign arguments to variables
DATASET_NAME="$1"
SUBSET_PATH="$2"
OUTPUTS_PATH="$3"
DATASET_CONFIG="$4"
LR="$5"
FINETUNE_TYPE="$6"
BATCH_SIZE="$7"
CHECKPOINT_PATH="$8"
TRAINING_TASK="$9"

# One-liner echo for debugging
echo "Running with: Dataset=$DATASET_NAME | Subset=$SUBSET_PATH | Output=$OUTPUTS_PATH | Config=$DATASET_CONFIG | LR=$LR | Finetune=$FINETUNE_TYPE | Batch=$BATCH_SIZE | Checkpoint=$CHECKPOINT_PATH | Training_task=$TRAINING_TASK"

# Run the training script
python baselines/train_on_subset.py \
    --dataset_name "$DATASET_NAME" \
    --subset_path "$SUBSET_PATH" \
    --outputs_path "$OUTPUTS_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --lr "$LR" \
    --finetune_type "$FINETUNE_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --training_task "$TRAINING_TASK" \