import os
import torch
import sys
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from model_backbone import get_lora_model, get_model_processor, get_features
import torch.optim as optim
from tqdm import tqdm
import json
import timm
from utils import get_dataset, get_metrics, get_train_val_dl
from utils import get_metrics
import yaml
import pandas as pd
import pdb
from train_engine import TrainEngine

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")
        

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["FMoW","COOS","iWildCam","GeoDE","CropHarvest","AutoArborist","SelfDrivingCar","ReID"],
        default="iWildCam",
        help="Dataset name",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        required=True,
        help="subset uid path",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="dataset config",
    )
    parser.add_argument(
        "--lr",
        type=str,
        required=False,
        help="lr",
    )
    parser.add_argument(
        "--finetune_type",
        type=str,
        required=True,
        choices=["linear_probe","lora_finetune_vit","full_finetune_resnet50","full_finetune_resnet101"],
        help="which type of finetuning is being done",
    )
    parser.add_argument(
        '--outputs_path', 
        type=str,required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=50,
        help="num_epochs",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="base path to save model",
    )
    parser.add_argument(
        "--training_task",
        type=str,
        required=True,
        choices=["classification","regression","detection","reid"]
    )
    parser.add_argument(
        "--only_evaluate",
        action="store_true",
        help="whether to only evaluate the model",
        required=False,
        default=False
    )

    args = parser.parse_args()
    model, preprocess = get_model_processor(args.finetune_type)

    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    task_list = dataset_config[args.dataset_name]['task_list']
    
    # getting data
    dataset = get_dataset(dataset_name=args.dataset_name, 
                          split='train', 
                          subset_path=args.subset_path, 
                          transform=preprocess)
    if args.training_task == 'classification':
        # removing single value labels
        label_counts = dataset.data['label'].value_counts()
        single_instance_labels = label_counts[label_counts == 1].index
        dataset.data = dataset.data[~dataset.data['label'].isin(single_instance_labels)]
        dataset.data = dataset.data.reset_index(drop=True)
        dataset.labels = dataset.data['label']
        # getting new mapping
        unique_values = dataset.labels.unique()  # get unique values
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        print(args.dataset_name, dataset.data.columns)
        dataset.data['label'] = dataset.data['label'].map(value_to_index)
        dataset.data.dropna(subset=['label'],inplace=True)
        dataset.data = dataset.data.reset_index()
        dataset.labels=dataset.data['label']
    train_dataset, val_dataset, train_dataloader, val_dataloader, num_classes = get_train_val_dl(dataset=dataset, 
                                                                                                 batch_size=int(args.batch_size),
                                                                                                 training_task=args.training_task)

    # need to check if its happening on all the tasks or just one - set "task list" accordingly
    subset_filename=args.subset_path.split('/')[-1]
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
                              checkpoint_path=args.checkpoint_path)
    if not args.only_evaluate:
        model = train_engine.train()

    # load the best checkpoint for model
    checkpoint = torch.load(args.checkpoint_path)
    # temp fix, needs to be fixed in the future
    if args.only_evaluate and args.training_task == "classification":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state"])

    # iterate through task list 
    for task_name in task_list:
        # get relevant test dataset and remap indices
        test_dataset = get_dataset(dataset_name=args.dataset_name, 
                                   split=task_name, 
                                   subset_path=None, 
                                   transform=preprocess)
        if args.training_task == 'classification':
            test_dataset.data['label'] = test_dataset.data['label'].map(value_to_index)
            test_dataset.data.dropna(subset=['label'],inplace=True)
            test_dataset.data=test_dataset.data.reset_index(drop=True)
            test_dataset.labels=test_dataset.data['label']
        print(f"testing on {task_name}", len(test_dataset))
        metrics, logits_dict = train_engine.evaluate(test_dataset=test_dataset, task_name=task_name)

        # save metrics
        with open(args.outputs_path+f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_metrics.json", "w") as json_file:
            metrics['subset_size']=len(train_dataset)
            json.dump(metrics, json_file, indent=4)
            torch.save({"logits": logits_dict['logits'], 
                        "labels": logits_dict['labels'], 
                        "preds": logits_dict['predictions'], 
                        "mapping": value_to_index}, 
                        f"{args.outputs_path}/{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_logits.pt")


if __name__ == "__main__":
    main()