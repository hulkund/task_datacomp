import os
import clip
import torch
import sys
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

import argparse
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from model_backbone import get_lora_model, get_model_processor, get_features
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import itertools
from pytorch_metric_learning import losses
import timm
# from utils import get_dataset, get_metrics, get_train_val_dl
from utils import get_metrics
import pandas as pd
import pdb


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("cuda")

    
def train_reid(model, 
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
    elif finetune_type in ["full_finetune_resnet50","full_finetune_resnet101","lora_finetune_vit"]:
        model.fc = torch.nn.LazyLinear(768)
        # get embedding size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            embedding_size = model(dummy_input).shape[1]
        criterion =  losses.ArcFaceLoss(num_classes=num_classes,
                                        embedding_size=embedding_size) # NEED TO FIX THIS
        params = itertools.chain(model.parameters(), criterion.parameters())
        optimizer = torch.optim.SGD(params=params, lr=0.001, momentum=0.9, weight_decay=0.0)
        model = train_full_finetune(model=model, 
                                    checkpoint_path=checkpoint_path,
                                    train_dataloader=train_dl,
                                    val_dataloader=val_dl, 
                                    num_epochs=num_epochs, 
                                    criterion=criterion, 
                                    optimizer=optimizer)
    return model

def evaluate_reid(model,
             test_dataset,
             dataset_name,
             preprocess,
             finetune_type: str = "linear_probe",
             task_name: str = "",
             batch_size: int = 32):
    if finetune_type=="linear_probe":
        test_features, test_labels = get_features(dataset_name=dataset_name, split=task_name, subset_path=None)
        predictions = model.predict(test_features)
        metrics = get_metrics(predictions=predictions, ground_truth=test_labels)
    else:
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
        metrics = evaluate_full_finetune(model, test_dataloader)
    return metrics

def evaluate_full_finetune(model, 
                           test_dataloader):
    device="cuda"
    model.eval()
    correct = 0
    total = 0
    embeddings_all = []
    targets_all = []
    encounter_idx_all = []
    with torch.no_grad():
        for data in test_dataloader:
            images, _, labels, _, encounter_idx = data
            print(images.shape)
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            embeddings_all.append(embeddings.cpu())
            targets_all.append(labels.cpu())
            encounter_idx_all.extend(encounter_idx.cpu())
    # Convert to a tensor
    embeddings = torch.cat(embeddings_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    encounter_idx = torch.cat(encounter_idx_all, dim=0)

    # compute cosine similarities
    embeddings = F.normalize(embeddings)
    similarities = embeddings @ embeddings.T

    # mask similarities of images within a single encounter
    similarities[encounter_idx == encounter_idx[None, :].T] = float("-inf")
    
    # Save logits and labels for analysis
    _, topk = similarities.topk(5, dim=1)
    accuracy = (topk.eq(torch.argwhere(targets == targets)[:, None]).any(dim=1) * 1.).mean().item()
    metrics = {"accuracy": accuracy, "subset_size": len(targets)}
    logits_dict = {"logits": logits_tensor, "labels": labels_tensor, "predictions": pred_tensor}
    metrics = get_metrics(predictions=predicted_all, ground_truth=labels_all)
    metrics['accuracy']=correct / total
    return metrics, logits_dict

def train_full_finetune(model, 
                        train_dataloader, 
                        val_dataloader, 
                        checkpoint_path,
                        num_epochs, 
                        criterion, 
                        optimizer,
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
    
    for epoch in range(resume_epoch,num_epochs):
        model.train()
        running_loss = 0.0
        # training loop
        for i, data in enumerate(tqdm(train_dataloader, 0)):
            inputs, _, labels, _, encounter_idx = data
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
                images, _, labels, _, encounter_idx = data
                images, labels = images.to(device), labels.to(device)
                embeddings = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
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
        print(f"Epoch {epoch + 1} validation loss: {val_loss / len(val_dataloader):.3f}, ")
        #f"accuracy: {100 * correct / total:.2f}%")
    model.load_state_dict(best_model_wts)
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
#     parser.add_argument('--outputs_path', type=str,required=True)
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
#     args = parser.parse_args()

#     model_init, preprocess = get_model_processor(args.finetune_type)

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
    
#     # training 
#     trained_model = train(train_dl=train_dataloader,
#                           val_dl=val_dataloader,
#                           finetune_type=args.finetune_type,
#                           num_epochs=int(args.num_epochs),
#                           batch_size=int(args.batch_size),
#                           dataset_name=args.dataset_name,
#                           lr=float(args.lr),
#                           model=model_init,
#                           preprocess=preprocess,
#                           subset_path=args.subset_path,
#                           num_classes=num_classes)

#     # testing, need to check if its happening on all the tasks or just one
#     subset_filename=args.subset_path.split('/')[-1]
#     if 'task' in subset_filename or 'test' in subset_filename:
#         task_name = subset_filename.split('_')[0]
#         task_list = [task_name]
#         torch.save(trained_model.state_dict(), args.outputs_path+f"{task_name}_{args.finetune_type}_lr={int(args.lr)}_batchsize={int(args.batch_size)}_model")
#     if 'all' in subset_filename:
#         torch.save(trained_model.state_dict(), args.outputs_path+f"all_{args.finetune_type}_lr={int(args.lr)}_batchsize={int(args.batch_size)}_model")

#     # iterate through task list 
#     for task_name in task_list:
#         # get relevant test dataset and remap indices
#         test_dataset = get_dataset(dataset_name=args.dataset_name, 
#                                    split=task_name, 
#                                    subset_path=None, 
#                                    transform=preprocess)
#         test_dataset.data['label'] = test_dataset.data['label'].map(value_to_index)
#         test_dataset.data.dropna(subset=['label'],inplace=True)
#         test_dataset.data=dataset.data.reset_index()
#         test_dataset.labels=test_dataset.data['label']
#         print(f"testing on {task_name}")
#         metrics, logits_dict = evaluate(model=trained_model, 
#                                        test_dataset=test_dataset, 
#                                        dataset_name=args.dataset_name,
#                                        preprocess=preprocess, 
#                                        finetune_type=args.finetune_type, 
#                                        task_name=task_name, 
#                                        batch_size=int(args.batch_size))
#         with open(args.outputs_path+f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_metrics.json", "w") as json_file:
#             json.dump(metrics, json_file, indent=4)
#             torch.save({"logits": logits_dict['logits'], "labels": logits_dict['labels'], "preds": logits_dict['predictions'], "mapping": value_to_index}, 
#                         f"{task_name}_{args.finetune_type}_lr={args.lr}_batchsize={args.batch_size}_logits.pt")


if __name__ == "__main__":
    main()