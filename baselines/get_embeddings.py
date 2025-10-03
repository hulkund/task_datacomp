import sys
import os
from utils import get_dataset
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
from transformers import CLIPProcessor, CLIPModel
import timm
import json
import torch
import clip

def fix_embedding(embedding):
    dict_embed = {}
    for e in embedding:
        for key, value in e.items():
            if key not in dict_embed:
                dict_embed[key] = []
            dict_embed[key].append(value)
    return dict_embed

    
def get_clip_embeddings(data_loader, clip_processor, clip_model):
    embeddings_list = []
    for images, texts, labels, uids in data_loader:
        image_inputs = clip_processor(images=images, padding=True, return_tensors="pt")
        text_inputs = clip_processor(text=texts, padding=True, return_tensors="pt")

        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            text_features = clip_model.get_text_features(**text_inputs)
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity_scores = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings, dim=-1)
        
        for img_emb, txt_emb, similarity_score, texts, uids, labels in zip(image_embeddings, text_embeddings, similarity_scores, texts, uids, labels):
            embeddings_list.append({"image_embedding": img_emb.numpy(), 
                               "text_embedding": txt_emb.numpy(), 
                               "similarity_score": similarity_score.numpy(),
                               "text": texts, 
                               "uid": uids,
                               "label": labels})
            
    return embeddings_list

def get_dino_embeddings(data_loader, preprocess, model):
    embeddings_list = []
    for images, texts, labels, uids in data_loader:
        # image_inputs = preprocess(images)

        with torch.no_grad():
            image_features = model(images)
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
        
        for img_emb, texts, uids, labels in zip(image_embeddings, texts, uids, labels):
            embeddings_list.append({"image_embedding": img_emb.numpy(),
                                    "text_embedding": None,  # DINO does not use text embeddings
                                    "similarity_score": None,  # No similarity score for DINO
                                    "text": texts,
                                    "uid": uids,
                                    "label": labels})
            
    return embeddings_list

def get_embedding(embedding_type, dataset_name, split):
    if embedding_type == "clip":
        model_name = "openai/clip-vit-base-patch32"
        preprocess = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        dataset = get_dataset(dataset_name=args.dataset_name,
                                 split=args.split,
                                subset_path=None)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        embeddings = get_clip_embeddings(data_loader, preprocess, model)
    elif embedding_type == "dino":
        model = timm.create_model(
            'vit_small_patch16_224.dino',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = model.eval()
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        dataset = get_dataset(dataset_name=args.dataset_name,
                                 split=args.split,
                                subset_path=None,
                                transform=preprocess)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        embeddings = get_dino_embeddings(data_loader, preprocess, model)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    return embeddings

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=["FMoW","COOS","iWildCam","GeoDE", "AutoArborist", "SelfDrivingCar", "iWildCamCropped", "CropHarvest", "FishDetection", "ReID"],
    default="COOS",
    help="Dataset name",
)
parser.add_argument(
    "--split",
    type=str,
    required=True,
    help="split",
)
parser.add_argument(
    "--embedding_type",
    type=str,
    choices=["clip", "dino"],
    required=True,
    help="dino, clip",
)
parser.add_argument(
    "--save_path",
    type=str,
)

args = parser.parse_args()

if not os.path.exists(args.save_path):
    print(f"Getting {args.embedding_type} embeddings for {args.split} split of {args.dataset_name} dataset")
    embeddings = get_embedding(args.embedding_type, args.dataset_name, args.split)
    # Convert to numpy array
    np.savez(args.save_path, embeddings=embeddings)

