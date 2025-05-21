import os
import clip
import torch
import sys
# sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.models import resnet50, ResNet50_Weights

import argparse
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from tqdm import tqdm
import json
from utils import get_dataset
import pdb
import yaml
import pdb
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, mean_squared_error
from train_on_subset import get_metrics, get_train_val_dl, get_model_processor, evaluate_full_finetune, train_full_finetune, get_features

torch.manual_seed(42)


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        default='iWildCam',
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        required=False,
        default=None,
        help="subset uid path, should be npy file",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=False,
        default='configs/datasets.yaml',
        help="should be yaml file",
    )
    parser.add_argument(
        "--lr",
        type=str,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        '--outputs_path', 
        type=str,
        required=True)
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=10,
        help="how many epochs before early stopping",
    )
    parser.add_argument(
        "--freeze_layers",
        type=int,
        required=False,
        default=False,
        help="which type of finetuning is being done",
    )
    
    args = parser.parse_args()

    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    task_list = dataset_config[args.dataset_name]['task_list']
    
    # evaluation, checking across all tasks
    for task in task_list:
        task_num = task[4]
        # train_features, train_labels = get_features(dataset_name=args.dataset_name, subset_path=None, split=f"val{task_num}")
        # if args.dataset_name == 'SelfDrivingCar':
        #     classifier = LinearRegression()
        # else:
        #     classifier = LogisticRegression(random_state=0, C=0.75, max_iter=1000, verbose=1)
        # classifier.fit(train_features, train_labels)
        # test_features, test_labels = get_features(dataset_name=args.dataset_name, split=task, subset_path=None)
        # predictions = classifier.predict(test_features)
        # if args.dataset_name == 'SelfDrivingCar':
        #     metrics=mean_squared_error(y_true=test_labels, y_pred=predictions)
        # else:
        #     metrics = get_metrics(predictions=predictions, ground_truth=test_labels)
        # print(metrics)
        # with open(args.outputs_path+f"/{task}_queryset_linearprobe_metrics.json", "w") as json_file:
        #     json.dump(metrics, json_file, indent=4)
        #getting model and data
        model, preprocess = get_model_processor(finetune_type="lora_finetune_vit")
        task_num = task[4]
        dataset = get_dataset(dataset_name=args.dataset_name, split=f"val{task_num}", subset_path=None, transform=preprocess)
        unique_values = dataset.labels.unique()  # Get unique values
        value_to_index = {value: idx for idx, value in enumerate(unique_values)}
        dataset.labels = dataset.labels.map(value_to_index)
        train_dataset, val_dataset, train_dataloader, val_dataloader, num_classes = get_train_val_dl(dataset=dataset, batch_size=int(args.batch_size))
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
        model = train_full_finetune(model, 
                                    train_dataloader, 
                                    val_dataloader, 
                                    int(args.num_epochs), 
                                    criterion, 
                                    optimizer, 
                                    int(args.patience))
        
        metrics = evaluate_full_finetune(model, test_dataloader)

        #can change the name of output file here
        with open(args.outputs_path+f"queryset_lr={args.lr}_batchsize={args.batch_size}_{task}_metrics.json", "w") as json_file:
            json.dump(metrics, json_file, indent=4)
        
            

if __name__ == "__main__":
    main()