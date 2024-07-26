import sys
from transformers import AutoImageProcessor, AutoModelForImageClassification
import transformers
import timm
import torch
import peft
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from utils import get_dataset, get_metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from torch.utils.data import random_split, DataLoader
from peft import get_peft_model, LoraConfig, PeftModel
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import json

class ViTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ViTModel(PreTrainedModel):
    config_class = ViTConfig

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.blocks = model.blocks

    def forward(self, x):
        return self.model(x)

def get_lora_model(model):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.01,
        bias='none',
        target_modules=['qkv'],
        modules_to_save=["classifier"],
    )
    extractor_model = get_peft_model(ViTModel(model, ViTConfig()), config).to('cuda')
    return extractor_model

def train(model, train_dataloader, val_dataloader, num_epochs, criterion, optimizer):
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
            # if i % 100 == 0:  # Print every 100 mini-batches
            #     print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            #     running_loss = 0.0
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
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

def evaluate(model, test_dataloader):
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


def get_train_val_dl(dataset_name, subset_path, preprocess, batch_size):
    train_dataset = get_dataset(dataset_name=dataset_name,
                                split="train",
                                subset_path=subset_path,
                                transform=preprocess)
    train_size = int(0.9 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, test_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=1, shuffle=False)
    return train_dataloader, val_dataloader

def main(args):
    # fix hyperparameters
    batch_size=args.batch_size
    torch.manual_seed(42) 
    lr=args.lr
    num_epochs=args.num_epochs

    # create model
    timm_model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
    model = get_lora_model(timm_model)

    # set optimizer, criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # get preprocessing
    data_config = timm.data.resolve_model_data_config(model)
    preprocess_train = timm.data.create_transform(**data_config, is_training=True, no_aug=True)
    preprocess_eval = timm.data.create_transform(**data_config, is_training=False)

    # train and save model
    train_dataloader, val_dataloader = get_train_val_dl(dataset_name=args.dataset_name, 
                                                        subset_path=args.subset_path, 
                                                        preprocess=preprocess_train, 
                                                        batch_size=batch_size)
    model = train(model=model,
                  train_dataloader=train_dataloader, 
                  val_dataloader=val_dataloader, 
                  num_epochs=num_epochs, 
                  criterion=criterion,
                  optimizer=optimizer)
    # get subset path and tasks
    subset_path=args.subset_path.split('/')[-1]
    if 'test' in subset_path:
        tasks = [subset_path[:5]]
    else:
        tasks=["test1","test2","test3","test4"]

    # evaluate model
    for task_name in tasks:
        torch.save(model.state_dict(), args.model_path +f"{task_name}_lora_model.pt")
        test_dataset = get_dataset(dataset_name=args.dataset_name, split=task_name, subset_path=None, transform=preprocess_eval)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1)
        metrics = evaluate(model=model, test_dataloader=test_dataloader)
        with open(args.outputs_path+f"{task_name}_lora_metrics.json", "w") as json_file:
            json.dump(metrics, json_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["FMoW","COOS","iWildCam"],
        help="Dataset name",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        required=True,
        help="subset uid path",
    )
    parser.add_argument(
        '--outputs_path', 
        type=str,
        required=True
    )
    parser.add_argument(
        '--num_epochs', 
        type=int,
        required=False,
        default=8
    )
    parser.add_argument(
        '--lr', 
        type=float,
        required=False,
        default=0.0001
    )
    parser.add_argument(
        '--seed', 
        type=int,
        required=False,
        default=42
    )
    parser.add_argument(
        '--batch_size', 
        type=int,
        required=False,
        default=256
    )
    
    args = parser.parse_args()
    args.model_path = args.outputs_path
    main(args)

