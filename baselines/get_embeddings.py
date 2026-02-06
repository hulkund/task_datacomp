import os
from utils import get_dataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
    
# CLIP-specific constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224),              # Resize shortest side
    transforms.CenterCrop(224),          # Ensure 224x224 square
    transforms.ToTensor(),               # Convert to [0, 1]
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD) # Shift to CLIP distribution
])

def get_clip_embeddings(data_loader, clip_processor, clip_model):
    print(f"get_clip_embedding with CLIP_TRANSFORM={CLIP_TRANSFORM} and no clip_processor for image")
    embeddings_list = []
    for images, texts, labels, uids in data_loader:
        image_inputs = {"pixel_values": images.to(clip_model.device)}
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

def get_embedding(embedding_type, dataset_name, split):
    if embedding_type == "clip":
        model_name = "openai/clip-vit-base-patch32"
        preprocess = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        dataset = get_dataset(dataset_name=dataset_name,
                              split=split,
                              subset_path=None,
                              transform=CLIP_TRANSFORM)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        embeddings = get_clip_embeddings(data_loader, preprocess, model)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    return embeddings

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=["iWildCam", "GeoDE", "AutoArborist"],
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
    choices=["clip"],
    required=True,
    help="clip",
)
parser.add_argument(
    "--save_path",
    type=str,
)

args = parser.parse_args()

if not os.path.exists(args.save_path):
    print(f"Getting {args.embedding_type} embeddings for {args.split} split of {args.dataset_name} dataset")
    embeddings = get_embedding(args.embedding_type, args.dataset_name, args.split)
    np.savez(args.save_path, embeddings=embeddings)
    print(f"Done. Embeddings saved to {args.save_path}")

