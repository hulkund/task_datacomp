import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from training.model_backbone import get_lora_model, get_model_processor, get_features
from baselines.utils import get_dataset, get_metrics, get_train_val_dl
import yaml
import subprocess
import time

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
     
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
          batch_size: int = 128,
          seed: int = 42,
          wandb_run=None):
    
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
                                    checkpoint_path=checkpoint_path,
                                    wandb_run=wandb_run)
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
    overall_acc = correct / total if total else 0.0
    metrics['accuracy'] = overall_acc
    metrics['class_avg_accuracy'] = metrics['class_avg_accuracy']

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
                        patience=5,
                        wandb_run=None):
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
        val_accuracy = 100 * correct / total if total else 0.0
        print(f"Epoch {epoch} validation loss: {val_loss}, accuracy: {val_accuracy:.2f}%")
        if wandb_run:
            wandb_run.log({
                "train_loss": running_loss / len(train_dataloader),
                "val_loss": val_loss / len(val_dataloader),
                "val_accuracy": val_accuracy,
                "epoch": epoch,
            })
    return model
