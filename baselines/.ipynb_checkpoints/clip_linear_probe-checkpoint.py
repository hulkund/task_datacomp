import os
import clip
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from all_datasets.FMoW_dataset import FMoWDataset
from all_datasets.COOS_dataset import COOSDataset
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_processor():
    model, preprocess = clip.load('ViT-B/32', device)
    return model, preprocess

def get_dataset(dataset_name,split,subset_path,transform):
    if dataset_name == "COOS":
        dataset = COOSDataset(split=split,subset_path=subset_path,transform=transform)
    elif dataset_name == "FMoW":
        dataset = FMoWDataset(split=split,subset_path=subset_path,transform=transform)
    return dataset

def get_features(model,dataset):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images,_, labels,_ in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def train(model,train_dataset,C):
    train_features, train_labels = get_features(model,train_dataset)
    classifier = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate(model,classifier, test_dataset, task_name):
    test_features, test_labels = get_features(model, test_dataset)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100
    metrics = get_metrics(predictions=predictions,
                          ground_truth=test_labels,
                         task_name=task_name)
    return metrics

def get_metrics(predictions, ground_truth, task_name):
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro')
    recall = recall_score(ground_truth, predictions, average='macro')
    metrics = {"acc":acc, "precision": precision, "recall":recall}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        choices=["openai/clip-vit-base-patch32"],
        default="openai/clip-vit-base-patch32",
        help="CLIP model type",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["FMoW","COOS"],
        default="COOS",
        help="Dataset name",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        required=True,
        help="subset uid path",
    )
    # parser.add_argument(
    #     "--C_value",
    #     type=str,
    #     required=False,
    #     default=0.25
    #     help="C_value",
    # )
    parser.add_argument('--outputs_path', type=str,required=True)
    args = parser.parse_args()
    print(f"training on {args.subset_path} for {args.dataset_name}")
    model, preprocess = get_model_processor()
    train_dataset = get_dataset(dataset_name=args.dataset_name,split="train",subset_path=args.subset_path,transform=preprocess)
    for C in [0.1,0.25,0.5,0.75]:
        classifier = train(model=model, train_dataset=train_dataset, C=C)
        for task_name in ["test1","test2","test3","test4"]:
            test_dataset = get_dataset(dataset_name=args.dataset_name,split=task_name,subset_path=None,transform=preprocess)
            # test_features, test_labels = get_features(model=model, dataset=test_dataset)
            metrics = evaluate(model=model, classifier=classifier, test_dataset=test_dataset, task_name=task_name)
            with open(args.outputs_path+f"{task_name}_C={str(C)}_metrics.json", "w") as json_file:
                json.dump(metrics, json_file, indent=4)
            

if __name__ == "__main__":
    main()