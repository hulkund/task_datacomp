import os
import clip
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')

import argparse
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json
from utils import get_dataset
import pdb
from utils import get_metrics

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")

def get_model_processor():
    model, preprocess = clip.load('ViT-B/32', device)
    return model, preprocess

def get_features_from_scratch(model,dataset,modality):
    device="cuda"
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, texts, labels,_ in tqdm(DataLoader(dataset, batch_size=500)):
            if modality=="image":
                features = model.encode_image(images.to(device))
            elif modality=="text":
                texts=clip.tokenize(texts).to(device)
                features = model.encode_text(texts.to(device))
            all_features.append(features)
            all_labels.append(labels)
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

def get_features(dataset_name, split, subset_path):
    device="cuda"
    embed=np.load(f'/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{dataset_name}//embeddings/{split}_image_label_embed.npz')
    if subset_path:
        subset=np.load(subset_path,allow_pickle=True)
        id_to_index = {id_: idx for idx, id_ in enumerate(subset)}
        mask = [id_to_index[id_] for id_ in subset if id_ in id_to_index]
        features=embed['features'][mask]
        labels=embed['labels'][mask]
    else:
        features=embed['features']
        labels=embed['labels']
    return features, labels
    
def train(model,train_features, train_labels, C, modality):
    # train_features, train_labels = get_features(model,train_dataset, modality)
    classifier = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate(model,classifier, test_features, test_labels, task_name, modality):
    # test_features, test_labels = get_features(model, test_dataset, modality)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100
    print("task:", task_name, " accuracy:", accuracy)
    metrics = get_metrics(predictions=predictions,
                          ground_truth=test_labels)
    return metrics

# def get_metrics(predictions, ground_truth, task_name):
#     acc = accuracy_score(ground_truth, predictions)
#     precision = precision_score(ground_truth, predictions, average='macro')
#     recall = recall_score(ground_truth, predictions, average='macro')
#     metrics = {"acc":acc, "precision": precision, "recall":recall}
#     return metrics


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
        choices=["FMoW","COOS","iWildCam","CivilComments"],
        default="COOS",
        help="Dataset name",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        required=True,
        help="subset uid path",
    )
    parser.add_argument('--outputs_path', type=str,required=True)
    model, preprocess = get_model_processor()
    args = parser.parse_args()
    if args.dataset_name == "CivilComments":
        modality="text"
        preprocess=None
    else:
        modality="image"
    print(f"training on {args.subset_path} for {args.dataset_name} on {modality}")
        
    
    # train_dataset = get_dataset(dataset_name=args.dataset_name,
    #                             split="train",
    #                             subset_path=args.subset_path,
    #                             transform=preprocess)
    
    for C in [0.75]:
        train_features, train_labels = get_features(dataset_name=args.dataset_name, split='train', subset_path=args.subset_path)
        print("length of train_features:", len(train_features))
        classifier = train(model=model, train_features=train_features, train_labels=train_labels, C=C, modality=modality)
        subset_path=args.subset_path.split('/')[-1]
        if 'test' in subset_path:
            task_name = subset_path[:5]
            print(f"testing on {task_name}")
            test_features, test_labels = get_features(dataset_name=args.dataset_name, split=task_name, subset_path=None)
            # test_dataset = get_dataset(dataset_name=args.dataset_name,
            #                            split=task_name,
            #                            subset_path=None,
            #                            transform=preprocess)
            metrics = evaluate(model=model, 
                               classifier=classifier, 
                               test_features=test_features,
                               test_labels=test_labels,
                               task_name=task_name,
                               modality=modality)
            metrics['subset_size']=len(train_features)
            with open(args.outputs_path+f"{task_name}_C={str(C)}_metrics.json", "w") as json_file:
                json.dump(metrics, json_file, indent=4)
        else:
            for task_name in ["test1","test2","test3","test4"]:
                test_features, test_labels = get_features(dataset_name=args.dataset_name, split=task_name, subset_path=None)
                # test_dataset = get_dataset(dataset_name=args.dataset_name,
                #                            split=task_name,
                #                            subset_path=None,
                #                            transform=preprocess)
                metrics = evaluate(model=model, 
                                   classifier=classifier, 
                                   test_features=test_features,
                                   test_labels=test_labels,
                                   task_name=task_name,
                                   modality=modality)
                metrics['subset_size']=len(train_features)
                with open(args.outputs_path+f"{task_name}_C={str(C)}_metrics.json", "w") as json_file:
                    json.dump(metrics, json_file, indent=4)
            

if __name__ == "__main__":
    main()