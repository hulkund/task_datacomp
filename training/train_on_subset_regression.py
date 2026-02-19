import os
import torch
import sys
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from training.model_backbone import get_lora_model, get_model_processor, get_features
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, random_split, Subset


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")
      
def train_regression(model,
          train_dl,
          val_dl,
          dataset_name,
          preprocess,
          subset_path,
          checkpoint_path,
          finetune_type: str = "full_finetune_resnet50",
          num_epochs: int = 30,
          lr: float = 0.01,
          C: float = 0.75,
          batch_size: int = 128,
          num_classes=1,
          seed: int = 42,
          wandb_run=None):
    if finetune_type=="linear_probe":
        train_features, train_labels = get_features(dataset_name=dataset_name, subset_path=subset_path, split='train')
        pca = PCA(n_components=512)
        train_features = pca.fit_transform(train_features)
        classifier = LinearRegression()
        classifier = classifier.fit(X=train_features, y=train_labels)
        return {"classifier":classifier,"pca":pca}
    elif finetune_type in ["full_finetune_resnet50","full_finetune_resnet101","lora_finetune_vit"]:
        if not finetune_type=="lora_finetune_vit": 
            model.fc = nn.Linear(model.fc.in_features, num_classes)        
        criterion = nn.MSELoss()
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

def evaluate_regression(model,
             test_dataset,
             dataset_name,
             preprocess,
             finetune_type: str = "linear_probe",
             task_name: str = "",
             batch_size: int = 32):
    if finetune_type=="linear_probe":
        test_features, test_labels = get_features(dataset_name=dataset_name, split=task_name, subset_path=None)
        classifier, pca = model["classifier"], model["pca"]
        test_features = pca.transform(test_features)
        predictions = classifier.predict(test_features)
        metrics = {'mse':str(mean_squared_error(y_pred=predictions, y_true=test_labels))}
    else:
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
        metrics = evaluate_full_finetune(model, test_dataloader)
    return metrics

def evaluate_full_finetune(model, test_dataloader):
    device="cuda"
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    predicted_all = []
    labels_all = []
    with torch.no_grad():
        for data in test_dataloader:
            images,_, labels,_ = data
            images, labels =  images.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            outputs = model(images).squeeze()
            predicted_all.extend(outputs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    metrics = {'mse':str(mean_squared_error(y_pred=predicted_all, y_true=labels_all))}
    return metrics 

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
    best_val_loss=np.inf
    patience_counter=0
    best_epoch = -1
    best_model_wts = None

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
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # training loop
        for i, data in enumerate(tqdm(train_dataloader, 0)):
            inputs, _, labels, _ = data
            inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
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
                images, labels = images.to(torch.float32).to(device), labels.to(torch.float32).to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        # check for early stopping
        if val_loss < best_val_loss:
            best_val_loss=val_loss
            patience_counter=0
            best_epoch = epoch
            best_model_wts = model.state_dict().copy()
        else:
            patience_counter+=1
        if patience_counter>=patience:
            print(f"early stopping at epoch {epoch}")
            break
        print(f"Epoch {epoch + 1} validation loss: {val_loss / len(val_dataloader):.3f}")
        if wandb_run:
            wandb_run.log({
                "train_loss": running_loss / len(train_dataloader),
                "val_loss": val_loss / len(val_dataloader),
                "epoch": epoch,
            })
    model.load_state_dict(best_model_wts)
    return model         

if __name__ == "__main__":
    main()