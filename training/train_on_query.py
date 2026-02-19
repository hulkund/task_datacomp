import os
import clip
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import json
from baselines.utils import get_dataset
import yaml
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, mean_squared_error
from training.train_on_subset import get_metrics, get_train_val_dl, get_model_processor, evaluate_full_finetune, train_full_finetune, get_features

# Set random seed for reproducibility
torch.manual_seed(42)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate model on query tasks.")
    parser.add_argument("--dataset_name", type=str, default='iWildCam')
    parser.add_argument("--subset_path", type=str, default=None, help="subset uid path, should be npy file")
    parser.add_argument("--dataset_config", type=str, default='configs/datasets.yaml', help="should be yaml file")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--outputs_path', type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10, help="how many epochs before early stopping")
    parser.add_argument("--freeze_layers", type=int, default=False, help="which type of finetuning is being done")
    args = parser.parse_args()

    # Load dataset config
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    task_list = dataset_config[args.dataset_name]['task_list']

    # Loop over all tasks in the task list
    for task in task_list:
        task_num = task[4]
        # Get model and preprocessing for the current task
        model, preprocess = get_model_processor(finetune_type="lora_finetune_vit")
        # Load and preprocess dataset for training/validation
        dataset = get_dataset(dataset_name=args.dataset_name, split=f"val{task_num}", subset_path=None, transform=preprocess)
        # Map labels to indices for consistency
        unique_values = dataset.labels.unique()
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        dataset.labels = dataset.labels.map(value_to_index)
        # Split into train/val sets and get dataloaders
        train_dataset, val_dataset, train_dataloader, val_dataloader, num_classes = get_train_val_dl(dataset=dataset, batch_size=int(args.batch_size))
        # Load and preprocess test dataset
        test_dataset = get_dataset(dataset_name=args.dataset_name, split=task, subset_path=None, transform=preprocess)
        test_dataset.labels = test_dataset.labels.map(value_to_index)
        test_dataset.data = test_dataset.data.dropna(subset=['label'])
        test_dataset.data = test_dataset.data.reset_index()
        test_dataloader = DataLoader(dataset=test_dataset, 
                                     batch_size=int(args.batch_size), 
                                     num_workers=1)
        # training 
        #model.fc = nn.Linear(model.fc.in_features, len(dataset.labels))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
        # Train the model
        model = train_full_finetune(model, train_dataloader, val_dataloader, int(args.num_epochs), criterion, optimizer, int(args.patience))
        # Evaluate the model
        metrics = evaluate_full_finetune(model, test_dataloader)
        # Save metrics to output file
        output_file = os.path.join(args.outputs_path, f"queryset_lr={args.lr}_batchsize={args.batch_size}_{task}_metrics.json")
        with open(output_file, "w") as json_file:
            json.dump(metrics, json_file, indent=4)
        print(f"Saved metrics for task {task} to {output_file}")

if __name__ == "__main__":
    main()