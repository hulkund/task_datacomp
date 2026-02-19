#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --output=/data/vision/beery/scratch/hasic/projects/task_datacomp/slurm/slurm-%J.out
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=100G
<<<<<<< HEAD
#SBATCH --time=1-23:00:00
#SBATCH --chdir=/data/vision/beery/scratch/hasic/projects/task_datacomp
#SBATCH --requeue

source /data/vision/beery/scratch/hasic/bashrc
conda init
conda activate unlabeled_exp
 
=======
#SBATCH --time=15:00:00
#SBATCH --requeue

source /data/vision/beery/scratch/neha/bash.rc
conda init
conda activate datacomp
>>>>>>> master

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
WANDB_PROJECT="${10}"
WANDB_ENTITY="${11}"
WANDB_GROUP="${12}"
WANDB_RUN_NAME="${13}"
NUM_EPOCHS="${14}"

# One-liner echo for debugging
echo "Running with: Dataset=$DATASET_NAME | Train_csv_custom=$RELABELED_TRAIN_CSV | Subset=$SUBSET_PATH | Output=$OUTPUTS_PATH | Config=$DATASET_CONFIG | LR=$LR | Finetune=$FINETUNE_TYPE | Batch=$BATCH_SIZE | Checkpoint=$CHECKPOINT_PATH | Training_task=$TRAINING_TASK | Wandb_project=$WANDB_PROJECT | Wandb_entity=$WANDB_ENTITY | Num_epochs=$NUM_EPOCHS"

# Build optional wandb arguments
WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS="--wandb_project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_entity $WANDB_ENTITY"
    fi
    if [ -n "$WANDB_GROUP" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_group $WANDB_GROUP"
    fi
    if [ -n "$WANDB_RUN_NAME" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

# Run the training script
python training/train_on_subset.py \
    --dataset_name "$DATASET_NAME" \
    --subset_path "$SUBSET_PATH" \
    --outputs_path "$OUTPUTS_PATH" \
    --dataset_config "$DATASET_CONFIG" \
    --lr "$LR" \
    --finetune_type "$FINETUNE_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --training_task "$TRAINING_TASK" \
    ${NUM_EPOCHS:+--num_epochs "$NUM_EPOCHS"} \
    $WANDB_ARGS