import os
import clip
import torch
import sys
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.models import resnet50, ResNet50_Weights

import argparse
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import json
from utils import get_dataset
import pdb
from utils import get_metrics
import yaml
import pdb

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")

def get_train_val_dl(dataset_name, subset_path, preprocess, batch_size):
    dataset = get_dataset(dataset_name=dataset_name,
                            split="train",
                            subset_path=subset_path,
                            transform=preprocess)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    train_labels = torch.tensor(dataset.labels)[train_dataset.indices]
    num_classes = train_labels.unique().numel()
    return train_dataloader, val_dataloader, num_classes

def get_model_processor(finetune_type):
    if finetune_type=="linear_probe":
        model, preprocess = clip.load('ViT-B/32', device)
        return model, preprocess
    elif finetune_type=="full_finetune":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        preprocess = weights.transforms()
        return model, preprocess

def get_features(dataset_name, split, subset_path):
    device="cuda"
    if 'task' in split:
        split='test'+split[4]
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
    
def train(model, 
          preprocess,
          subset_path,
          dataset_name: str = "iWildCam",
          finetune_type: str = "linear_probe",
          num_epochs: int = 30, 
          lr: float = 0.01,
          C: float = 0.75,
          batch_size: int = 32):
    if finetune_type=="linear_probe":
        train_features, train_labels = get_features(dataset_name=dataset_name, subset_path=subset_path, split='train')
        classifier = LogisticRegression(random_state=0, C=0.75, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)
        return classifier
    elif finetune_type == "full_finetune":
        if finetune_type=="lora_full_finetune": model = model.lora()
        train_dataloader, val_dataloader, num_classes = get_train_val_dl(dataset_name=dataset_name, 
                                                        subset_path=subset_path, 
                                                        preprocess=preprocess, 
                                                        batch_size=batch_size)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_full_finetune(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer)
    return model

def evaluate(model,
             preprocess,
             dataset_name: str = "iWildCam",
             finetune_type: str = "linear_probe",
             task_name: str = "",
             batch_size: int = 32):
    if finetune_type=="linear_probe":
        test_features, test_labels = get_features(dataset_name=dataset_name, split=task_name, subset_path=None)
        predictions = model.predict(test_features)
        metrics = get_metrics(predictions=predictions, ground_truth=test_labels)
    else:
        test_dataset = get_dataset(dataset_name=dataset_name, split=task_name, subset_path=None, transform=preprocess)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1)
        metrics = evaluate_full_finetune(model, test_dataloader)
    return metrics

def evaluate_full_finetune(model, test_dataloader):
    device="cuda"
    model.eval()
    correct = 0
    total = 0
    predicted_all = []
    labels_all = []
    with torch.no_grad():
        for data in test_dataloader:
            images,_, labels,_ = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_all.extend(predicted.cpu())
            labels_all.extend(labels.cpu())
    metrics = get_metrics(predictions=predicted_all, ground_truth=labels_all)
    return metrics 

def train_full_finetune(model, 
                        train_dataloader, 
                        val_dataloader, 
                        num_epochs, 
                        criterion, 
                        optimizer):
    device='cuda'
    model.to(device)
    best_val_loss=np.inf
    patience=2
    patience_counter=0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # training loop
        for i, data in tqdm(enumerate(train_dataloader, 0),total=len(train_dataloader)):
            inputs, _, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for data in val_dataloader:
                images, _,  labels, _ = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # check for early stopping
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            patience_counter=0
        else:
            patience_counter+=1
        if patience_counter>=patience:
            print(f"early stopping at epoch {epoch}")
            break
        print(f"Epoch {epoch + 1} validation loss: {val_loss / len(val_dataloader):.3f}, "
          f"accuracy: {100 * correct / total:.2f}%")
    return model

def main():
    parser = argparse.ArgumentParser(description="")
    
    # parser.add_argument(
    #     "--model_name",
    #     type=str,
    #     required=False,
    #     choices=["openai/clip-vit-base-patch32"],
    #     default="openai/clip-vit-base-patch32",
    #     help="CLIP model type",
    # )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["FMoW","COOS","iWildCam","GeoDE","CropHarvest","AutoArborist","SelfDrivingCar"],
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
        choices=["linear_probe","lora_full_finetune","full_finetune"],
        help="which type of finetuning is being done",
    )
    parser.add_argument('--outputs_path', type=str,required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="which type of finetuning is being done",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=30,
        help="which type of finetuning is being done",
    )
    args = parser.parse_args()

    model_init, preprocess = get_model_processor(args.finetune_type)

    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    task_list = dataset_config[args.dataset_name]['task_list']
    
    # training
    trained_model = train(dataset_name=args.dataset_name,
          finetune_type=args.finetune_type,
          num_epochs=int(args.num_epochs),
          batch_size=int(args.batch_size),
          lr=float(args.lr),
          model=model_init,
          preprocess=preprocess,
          subset_path=args.subset_path)
    
    # testing, need to check if its happening on all the tasks or just one
    subset_filename=args.subset_path.split('/')[-1]
    if 'task' in subset_filename or 'test' in subset_filename:
        task_name = subset_filename.split('_')[0]
        print(f"testing on {task_name}")
        metrics = evaluate(model=trained_model, dataset_name=args.dataset_name, preprocess=preprocess, finetune_type=args.finetune_type, task_name=task_name, batch_size=int(args.batch_size))
        with open(args.outputs_path+f"{task_name}_{args.finetune_type}_lr={args.lr}_metrics.json", "w") as json_file:
            json.dump(metrics, json_file, indent=4)
    else:
        for task_name in task_list:
            metrics = evaluate(model=trained_model, dataset_name=args.dataset_name, finetune_type=args.finetune_type, preprocess=preprocess, task_name=task_name, batch_size=int(args.batch_size))
            with open(args.outputs_path+f"{task_name}_{args.finetune_type}_lr={args.lr}_metrics.json", "w") as json_file:
                json.dump(metrics, json_file, indent=4)
            

if __name__ == "__main__":
    main()