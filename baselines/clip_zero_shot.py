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
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json
from utils import get_dataset
import pdb
import pandas as pd

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")

def get_model_processor():
    model, preprocess = clip.load('ViT-B/32', device)
    return model, preprocess

def get_predictions(model, dataset):
    device="cuda"

    labels_all = dataset.category_name
    text_tokens = clip.tokenize(labels_all).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    results = []
    dataloader = DataLoader(dataset, batch_size=500)
    with torch.no_grad():
        for batch_idx, (images, texts, labels, uids) in enumerate(dataloader):
            image_features = model.encode_image(images.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            best_match_indices = similarities.argmax(dim=1).cpu().numpy()
            for idx, best_match_idx in enumerate(best_match_indices):
                # pdb.set_trace()
                results.append({
                    'image_index': batch_idx * dataloader.batch_size + idx,
                    'predicted_label': labels_all[best_match_idx],
                    'label':dataset.mapping[labels[idx].item()]
                })
    df_results = pd.DataFrame(results)
    return df_results

# def evaluate(results):
#     for result in results:
        
#     test_features, test_labels = get_features(model, test_dataset, modality)
#     accuracy = np.mean((test_labels == predictions).astype(float)) * 100
#     metrics = get_metrics(predictions=predictions,
#                           ground_truth=test_labels,
#                          task_name=task_name)
#     return metrics

def get_metrics(predictions, ground_truth, task_name):
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro')
    recall = recall_score(ground_truth, predictions, average='macro')
    metrics = {"acc":acc, "precision": precision, "recall":recall}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["FMoW","COOS","iWildCam","CivilComments"],
        default="COOS",
        help="Dataset name",
    )

    args = parser.parse_args()
    outputs_path = f"/data/vision/beery/scratch/neha/task-datacomp/experiments/{args.dataset_name}/zero_shot/"
    
    print(f"training on {args.dataset_name}")
        
    model, preprocess = get_model_processor()
    for task_name in ["test1","test2","test3","test4"]:
        dataset = get_dataset(dataset_name=args.dataset_name, split=task_name, transform=preprocess)
        df_results = get_predictions(model, dataset)
        predictions = df_results['predicted_label']
        ground_truth = df_results['label']
        # print(predictions[:5], ground_truth[:5])
        metrics = get_metrics(predictions, ground_truth, task_name)
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)
        with open(outputs_path+f"{task_name}_metrics.json", "w") as json_file:
            json.dump(metrics, json_file, indent=4)
            

if __name__ == "__main__":
    main()