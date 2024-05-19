import numpy as np
# import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torchvision 
# from transformers import CLIPProcessor, CLIPModel
import timm
from all_datasets.COOS_dataset import COOSDataset
from all_datasets.FMoW_dataset import FMoWDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse


def save_model(model, epoch, save_path):
    model.save_pretrained(save_path+f"model.pt")
    # processor.save_pretrained(save_path+f"processor.pt")

def get_train_dataset(dataset_name, split, batch_size, subset_path):
    if dataset_name == "COOS":
        dataset = COOSDataset(split)
    elif dataset_name == "FMoW":
        dataset = FMoWDataset(split,transform=preprocess)
    # elif dataset_name == "iWildCam":
    #     continue #TODO
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, train_dataloader

def get_val_dataset(dataset_name, split, batch_size):
    if dataset_name == "COOS":
        dataset = COOSDataset('val1')
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, val_dataloader


def get_model_processor(num_classes):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.train()
    model.text_projection = torch.nn.Identity()
    model.vision_projection = torch.nn.Linear(in_features=512, out_features=num_classes)
    model = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
    return model, processor

# def get_model(num_classes):
#     model = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
#     model.classifier = nn.Linear(768, num_classes).to(device)  a linear layer for classification and move to GPU
    
#     return model
    

def train(train_dataloader, model, optimizer, num_epochs, device):
    loss_function = CrossEntropyLoss()
    model.to(device)
    model.train()
    for images, texts, labels, uids in train_dataloader:
        images= images.to(device)
        # texts=texts.to(device)
        # with torch.no_grad():
        #     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        # logits = model(**inputs).logits_per_image
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() 
        pred_label = torch.argmax(logits, dim=1)
        oa = torch.mean((pred_label == labels).float())
        a_total += oa.item()
    loss_total /= len(dataLoader)           
    oa_total /= len(dataLoader)
    return loss_total, oa_total

def validate(val_dataloader, model, device):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    model.to(device)
    model.eval()
    loss_function = nn.CrossEntropyLoss()   
    loss_total, oa_total = 0.0, 0.0     

    with torch.no_grad():               
        for data, texts, labels, uids in val_dataloader:
            data, labels, texts = data.to(device), labels.to(device), texts.to(device)
            # with torch.no_grad():
            #     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            logits = model(images)#.logits_per_image
            loss = loss_function(logits, labels)
            loss_total += loss.item()
            pred_label = torch.argmax(logits, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()
            progressBar.set_description(
                '[Val ] Loss: {:.4f}; OA: {:.4f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)
    progressBar.close()
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)
    return loss_total, oa_total

def main():
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    # parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for training', default=0.001)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=128)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training', default=200)
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--subset_path',type=str, help="subset filepath")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # loading things
    train_dataset, train_dataloader = get_train_dataset(dataset_name=args.dataset,
                                                         split="train",
                                                         batch_size=args.batch_size,
                                                        subset_path=args.subset_path)
    # NEED TO FIX
    val_dataset, val_dataloader = get_val_dataset(dataset_name=args.dataset,
                                                split=None,
                                                batch_size=args.batch_size)
    model = get_model(num_classes=train_dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    # early stopping
    patience = 3 
    early_stopping_counter = 0

    # setting variables
    best_val_loss = float('inf')
    best_epoch = None
    best_model = None
    num_epochs = args.num_epochs
    current_epoch=0

    while current_epoch < num_epochs:
        current_epoch += 1
        loss_train, oa_train = train(train_dataloader=train_dataloader,
                                    model=model,
                                    # processor=processor, 
                                    optimizer=optimizer,
                                    num_epochs=num_epochs,
                                    device=device)
        loss_val, oa_val = validate(val_dataloader=val_dataloader, 
                                    model=model, 
                                    device=device)
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_epoch = current_epoch
            best_model = model
            name = f'Best_Epoch_{current_epoch}_Loss_{round(loss_val, 2)}'
            early_stopping_counter = 0  
        else:
            early_stopping_counter += 1  
            if early_stopping_counter >= patience:
                print("Early stopping: Validation loss hasn't decreased for {} epochs.".format(patience))
                if best_val_loss < float('inf'):
                    save_model(model=best_model, save_path=args.model_save_path, epoch=best_epoch)

    if best_val_loss < float('inf'):
        save_model(model=best_model, save_path=args.model_save_path, epoch=best_epoch)

if __name__ == '__main__':
    main()

