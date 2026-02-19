"""
train_on_subset.py

Script for training and evaluating models on data subsets for various tasks (classification, regression, detection, reid).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')

import argparse
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.model_backbone import get_lora_model, get_model_processor, get_features
from baselines.utils import get_dataset, get_metrics, get_train_val_dl
from train_engine import TrainEngine
import wandb

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def main():
    """
    Main function to train and evaluate a model on a subset of data for a specified task.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate model on subset.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        choices=["iWildCam","GeoDE","CropHarvest","AutoArborist","SelfDrivingCar","ReID"],
                        default="iWildCam", help="Dataset name")
    parser.add_argument("--subset_path", type=str, required=False, help="subset uid path")
    parser.add_argument("--dataset_config", type=str, required=True, help="dataset config")
    parser.add_argument("--lr", type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument("--finetune_type", type=str, required=True,
                        choices=["linear_probe","lora_finetune_vit","full_finetune_resnet50","full_finetune_resnet101"],
                        help="Type of finetuning")
    parser.add_argument('--outputs_path', type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--num_epochs", type=int, required=False, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to save model")
    parser.add_argument("--training_task", type=str, required=True,
                        choices=["classification","regression","detection","reid"])
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    parser.add_argument("--only_evaluate", action="store_true", default=False, help="Whether to only evaluate the model")
    parser.add_argument("--relabeled_train_csv", type=str, required=False, default=None, help="Path to custom train CSV")
    parser.add_argument("--wandb_project", type=str, required=False, default=None, help="Wandb project name (enables wandb logging)")
    parser.add_argument("--wandb_entity", type=str, required=False, default=None, help="Wandb entity/team name")
    parser.add_argument("--wandb_group", type=str, required=False, default=None, help="Wandb group name for organizing related runs")
    parser.add_argument("--wandb_run_name", type=str, required=False, default=None, help="Wandb run name")
    args = parser.parse_args()

    # Initialize wandb if project name is provided
    wandb_run = None
    if args.wandb_project:
        run_name = args.wandb_run_name or f"{args.dataset_name}/{args.finetune_type}/lr={args.lr}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config={
                "dataset_name": args.dataset_name,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "finetune_type": args.finetune_type,
                "training_task": args.training_task,
                "subset_path": args.subset_path,
                "seed": args.seed,
            },
        )

    # Get model and preprocessing
    model, preprocess = get_model_processor(args.finetune_type)

    # Load dataset config and task list
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    task_list = dataset_config[args.dataset_name]['task_list']

    # Load and preprocess training dataset if not relabeled CSV provided
    if args.relabeled_train_csv:
        df = pd.read_csv(args.relabeled_train_csv)
        dataset = get_dataset(dataset_name=args.dataset_name, split='train', subset_path=None, transform=preprocess, dataframe=df)
    else:
        dataset = get_dataset(dataset_name=args.dataset_name, split='train', subset_path=args.subset_path, transform=preprocess)
    if args.training_task == 'classification':
        # Remove single-instance labels for classification
        label_counts = dataset.data['label'].value_counts()
        single_instance_labels = label_counts[label_counts == 1].index
        dataset.data = dataset.data[~dataset.data['label'].isin(single_instance_labels)]
        dataset.data = dataset.data.reset_index(drop=True)
        dataset.labels = dataset.data['label']
        # Remap labels to indices
        unique_values = dataset.labels.unique()
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        dataset.data['label'] = dataset.data['label'].map(value_to_index)
        dataset.data.dropna(subset=['label'], inplace=True)
        dataset.data = dataset.data.reset_index(drop=True)
        dataset.labels = dataset.data['label']
    # Split into train/val sets and get dataloaders
    train_dataset, val_dataset, train_dataloader, val_dataloader, num_classes = get_train_val_dl(
        dataset=dataset, batch_size=int(args.batch_size), training_task=args.training_task)

    # If subset filename indicates a specific task, update task_list
    if args.subset_path is not None:
        subset_filename = os.path.basename(args.subset_path)
        if 'task' in subset_filename or 'test' in subset_filename:
            task_name = subset_filename.split('_')[0]
            task_list = [task_name]

    # training 
    train_engine = TrainEngine(training_task=args.training_task,
                              train_dl=train_dataloader, 
                              val_dl=val_dataloader,
                              finetune_type=args.finetune_type,
                              num_epochs=int(args.num_epochs),
                              batch_size=int(args.batch_size),
                              dataset_name=args.dataset_name,
                              lr=float(args.lr),
                              model=model,
                              preprocess=preprocess,
                              subset_path=args.subset_path,
                              num_classes=num_classes,
                              checkpoint_path=args.checkpoint_path,
                              seed=args.seed,
                              wandb_run=wandb_run)
    if not args.only_evaluate:
        model = train_engine.train()

    # Load the best checkpoint for model
    checkpoint = torch.load(args.checkpoint_path)
    if args.only_evaluate and args.training_task == "classification":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])

    # Evaluate on each task in task_list
    for task_name in task_list:
        test_dataset = get_dataset(dataset_name=args.dataset_name, split=task_name, subset_path=None, transform=preprocess)
        if args.training_task == 'classification':
            test_dataset.data['label'] = test_dataset.data['label'].map(value_to_index)
            test_dataset.data.dropna(subset=['label'], inplace=True)
            test_dataset.data = test_dataset.data.reset_index(drop=True)
            test_dataset.labels = test_dataset.data['label']
        print(f"Testing on {task_name}, test set size: {len(test_dataset)}")
        metrics, logits_dict = train_engine.evaluate(test_dataset=test_dataset, task_name=task_name)
        # Save metrics and logits
        metrics['subset_size'] = len(train_dataset)
        if wandb_run:
            wandb_run.log({f"{args.dataset_name}/{task_name}/{k}": v for k, v in metrics.items()})
        metrics_path = os.path.join(args.outputs_path, f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_metrics.json")
        with open(metrics_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)
        torch.save({
            "logits": logits_dict['logits'],
            "labels": logits_dict['labels'],
            "preds": logits_dict['predictions'],
            "mapping": value_to_index if args.training_task == 'classification' else None
        }, os.path.join(args.outputs_path, f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_logits.pt"))

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
