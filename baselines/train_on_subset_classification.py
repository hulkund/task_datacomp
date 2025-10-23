import os
import torch
import sys
sys.path.append('/data/vision/beery/scratch/evelyn/task-datacomp/')
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

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")

     
def train_classification(model, 
          train_dl,
          val_dl,
          dataset_name,
          preprocess,
          subset_path,
          num_classes,
          checkpoint_path,
          finetune_type: str = "full_finetune",
          num_epochs: int = 30, 
          lr: float = 0.01,
          C: float = 0.75,
          batch_size: int = 128):
    
    if finetune_type=="linear_probe":
        train_features, train_labels = get_features(dataset_name=dataset_name, subset_path=subset_path, split='train')
        classifier = LogisticRegression(random_state=0, C=0.75, max_iter=1000, verbose=1)
        classifier.fit(train_features, train_labels)
        return classifier
    elif finetune_type in ["full_finetune_resnet50","full_finetune_resnet101","lora_finetune_vit"] :
        if not finetune_type=="lora_finetune_vit": 
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = train_full_finetune(model=model, 
                                    train_dataloader=train_dl,
                                    val_dataloader=val_dl, 
                                    num_epochs=num_epochs, 
                                    criterion=criterion, 
                                    optimizer=optimizer,
                                    checkpoint_path=checkpoint_path)
    return model


def evaluate_classification(
             test_dataset,
             task_name,
             model,
             dataset_name,
             preprocess,
             finetune_type: str = "linear_probe",
             batch_size: int = 32):
    if finetune_type=="linear_probe":
        test_features, test_labels = get_features(dataset_name=dataset_name, split=task_name, subset_path=None)
        predictions = model.predict(test_features)
        metrics = get_metrics(predictions=predictions, ground_truth=test_labels)
    else:
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=128, num_workers=8, shuffle=False, pin_memory=True)
        metrics = evaluate_full_finetune(model, test_dataloader)
    return metrics

# def evaluate_full_finetune(model, test_dataloader):
#     device="cuda"
#     model.eval()
#     model.to(device)
#     correct = 0
#     total = 0
#     predicted_all = []
#     labels_all = []
#     logits_all = []
#     print(device)
#     print(len(test_dataloader))
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_dataloader,0)):
#             images,_, labels,_ = data
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             logits_all.append(outputs.cpu())
#             predicted_all.append(predicted.cpu())
#             labels_all.append(labels.cpu())
#     # Convert to a tensor
#     logits_tensor = torch.cat(logits_all, dim=0)
#     labels_tensor = torch.cat(labels_all, dim=0)
#     pred_tensor = torch.cat(predicted_all, dim=0)
#     # Save logits and labels for analysis
#     logits_dict = {"logits": logits_tensor, "labels": labels_tensor, "predictions": pred_tensor}
#     metrics = get_metrics(predictions=predicted_all, ground_truth=labels_all)
#     metrics['accuracy']=correct / total
#     return metrics, logits_dict
import time
import subprocess

def print_gpu_utilization():
    try:
        # Query GPU memory and utilization
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        util, mem = result.stdout.strip().split(", ")
        print(f"🔹 GPU Utilization: {util}% | Memory Used: {mem} MB")
    except Exception as e:
        print(f"(GPU utilization check failed: {e})")


def evaluate_full_finetune(model, test_dataloader, log_gpu_every=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    logits_all = []
    predicted_all = []
    labels_all = []

    batch_times = []
    data_times = []

    start_total = time.time()
    end_data = time.time()

    with torch.no_grad():
        for i, (images, _, labels, _) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Measure data loading time
            data_time = time.time() - end_data
            data_times.append(data_time)

            # Move to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            start_batch = time.time()

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store tensors (append, not extend)
            logits_all.append(outputs.cpu())
            predicted_all.append(predicted.cpu())
            labels_all.append(labels.cpu())

            batch_time = time.time() - start_batch
            batch_times.append(batch_time)

            # # Periodically log GPU stats
            # if device == "cuda" and (i % log_gpu_every == 0):
            #     print_gpu_utilization()

            # Mark end of this iteration’s data loading for next iteration
            end_data = time.time()

    # Concatenate tensors in one go
    logits_tensor = torch.cat(logits_all, dim=0)
    labels_tensor = torch.cat(labels_all, dim=0)
    pred_tensor = torch.cat(predicted_all, dim=0)

    # Metrics
    metrics = get_metrics(predictions=pred_tensor, ground_truth=labels_tensor)
    metrics['accuracy'] = correct / total

    total_time = time.time() - start_total

    print(f"\n✅ Evaluation complete in {total_time:.2f} seconds")
    print(f"   Avg batch time: {sum(batch_times)/len(batch_times):.4f} s")
    print(f"   Avg data load time: {sum(data_times)/len(data_times):.4f} s")

    return metrics, {"logits": logits_tensor, "labels": labels_tensor, "predictions": pred_tensor}

def train_full_finetune(model, 
                        train_dataloader, 
                        val_dataloader, 
                        num_epochs, 
                        criterion, 
                        optimizer,
                        checkpoint_path,
                        patience=5):
    device='cuda'
    model.to(device)

    # check if resuming from a checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        resume_epoch = int(checkpoint["epoch"])
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_loss = float(checkpoint["best_val_loss"])
        patience_counter = int(checkpoint["patience_counter"])
        print(f"Resumed from epoch {resume_epoch}")
    else:
        resume_epoch = 0
        best_val_loss=np.inf
        patience_counter=0
        best_epoch = -1
        print("Training from scratch") 
    
    # training loop
    for epoch in range(resume_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        # training loop
        for i, data in enumerate(tqdm(train_dataloader, 0)):
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
            best_epoch = epoch 
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter":patience_counter
            }, checkpoint_path)
        else:
            patience_counter+=1
        if patience_counter>=patience:
            print(f"early stopping at epoch {epoch}")
            break
        print(f"Epoch {epoch} validation loss: {val_loss}, "f"accuracy: {100 * correct / total:.2f}%")
    return model

# def main():
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         required=True,
#         choices=["FMoW","COOS","iWildCam","GeoDE","CropHarvest","AutoArborist","SelfDrivingCar"],
#         default="iWildCam",
#         help="Dataset name",
#     )
#     parser.add_argument(
#         "--subset_path",
#         type=str,
#         required=True,
#         help="subset uid path",
#     )
#     parser.add_argument(
#         "--dataset_config",
#         type=str,
#         required=True,
#         help="dataset config",
#     )
#     parser.add_argument(
#         "--lr",
#         type=str,
#         required=False,
#         help="lr",
#     )
#     parser.add_argument(
#         "--finetune_type",
#         type=str,
#         required=True,
#         choices=["linear_probe","lora_finetune_vit","full_finetune","full_finetune_resnet101"],
#         help="which type of finetuning is being done",
#     )
#     parser.add_argument(
#         '--outputs_path', 
#         type=str,required=True)
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         required=False,
#         default=32,
#     )
#     parser.add_argument(
#         "--num_epochs",
#         type=int,
#         required=False,
#         default=50,
#         help="num_epochs",
#     )
#     parser.add_argument(
#         "--checkpoint_path",
#         type=str,
#         required=True,
#         help="base path to save model",
#     )
#     args = parser.parse_args()

#     model, preprocess = get_model_processor(args.finetune_type)

#     with open(args.dataset_config) as f:
#         dataset_config = yaml.safe_load(f)
#     task_list = dataset_config[args.dataset_name]['task_list']
    
#     # getting data
#     dataset = get_dataset(dataset_name=args.dataset_name, 
#                           split='train', 
#                           subset_path=args.subset_path, 
#                           transform=preprocess)
#     unique_values = dataset.labels.unique()  # Get unique values
#     value_to_index = {value: idx for idx, value in enumerate(unique_values)}
#     dataset.data['label'] = dataset.data['label'].map(value_to_index)
#     dataset.data.dropna(subset=['label'],inplace=True)
#     dataset.data=dataset.data.reset_index()
#     dataset.labels=dataset.data['label']
#     train_dataset, val_dataset, train_dataloader, val_dataloader, num_classes = get_train_val_dl(dataset=dataset, batch_size=int(args.batch_size))

#     # need to check if its happening on all the tasks or just one - set "task list" accordingly
#     subset_filename=args.subset_path.split('/')[-1]
#     if 'task' in subset_filename or 'test' in subset_filename:
#         task_name = subset_filename.split('_')[0]
#         task_list = [task_name]
    
#     # training  
#     train(train_dl=train_dataloader, 
#           val_dl=val_dataloader,
#           finetune_type=args.finetune_type,
#           num_epochs=int(args.num_epochs),
#           batch_size=int(args.batch_size),
#           dataset_name=args.dataset_name,
#           lr=float(args.lr),
#           model=model,
#           preprocess=preprocess,
#           subset_path=args.subset_path,
#           num_classes=num_classes,
#           checkpoint_path=args.checkpoint_path)  

#     # load the best checkpoint for model
#     checkpoint = torch.load(args.checkpoint_path)
#     model.load_state_dict(checkpoint["model_state"])
    
#     # iterate through task list 
#     for task_name in task_list:
#         # get relevant test dataset and remap indices
#         test_dataset = get_dataset(dataset_name=args.dataset_name, 
#                                    split=task_name, 
#                                    subset_path=None, 
#                                    transform=preprocess)
#         test_dataset.data['label'] = test_dataset.data['label'].map(value_to_index)
#         test_dataset.data.dropna(subset=['label'],inplace=True)
#         test_dataset.data=dataset.data.reset_index(drop=True)
#         test_dataset.labels=test_dataset.data['label']
#         print(f"testing on {task_name}")
#         metrics, logits_dict = evaluate(model=model, 
#                                        test_dataset=test_dataset, 
#                                        dataset_name=args.dataset_name,
#                                        preprocess=preprocess, 
#                                        finetune_type=args.finetune_type, 
#                                        task_name=task_name, 
#                                        batch_size=int(args.batch_size))

#         # save metrics
#         with open(args.outputs_path+f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_metrics.json", "w") as json_file:
#             metrics['subset_size']=len(train_dataset)
#             json.dump(metrics, json_file, indent=4)
#             torch.save({"logits": logits_dict['logits'], 
#                         "labels": logits_dict['labels'], 
#                         "preds": logits_dict['predictions'], 
#                         "mapping": value_to_index}, 
#                         f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_logits.pt")


# if __name__ == "__main__":
#     main()