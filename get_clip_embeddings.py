import torch
from transformers import CLIPProcessor, CLIPModel
from all_datasets.COOS_dataset import COOSDataset
from all_datasets.FMoW_dataset import FMoWDataset
from PIL import Image
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader


DATASET_NAMES = ["COOS","FMoW","iWildCam"]

# Load CLIP model and processor
def preprocess_images(images, clip_processor):
    image_input = clip_processor(images=images, padding=True, return_tensors="pt")
    return image_input

def preprocess_texts(texts, clip_processor):
    text_input = clip_processor(texts, padding=True, return_tensors="pt")
    return text_input

def get_dataloader(dataset_name,split):
    # ADD TO THIS
    if dataset_name == "COOS":
        dataset = COOSDataset(split)
    elif dataset_name == "FMoW":
        dataset = FMoWDataset(split)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    return dataloader
    
def get_embeddings(data_loader, clip_processor, clip_model):
    embeddings_list = []
    for images, texts, labels, uids in data_loader:
        image_inputs = preprocess_images(images, clip_processor)
        text_inputs = preprocess_texts(texts, clip_processor)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            text_features = clip_model.get_text_features(**text_inputs)
            image_embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
            text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity_scores = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings, dim=-1)
        
        for img_emb, txt_emb, similarity_score, texts, uids in zip(image_embeddings, text_embeddings, similarity_scores, texts, uids):
            embeddings_list.append({"image_embedding": img_emb.numpy(), 
                               "text_embedding": txt_emb.numpy(), 
                               "similarity_score": similarity_score.numpy(),
                               "text": texts, 
                               "uid": uids})
            
    return embeddings_list

def fix_embedding(embedding):
    dict_embed = {}
    for e in embedding:
        for key, value in e.items():
            if key not in dict_embed:
                dict_embed[key] = []
            dict_embed[key].append(value)
    return dict_embed

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
        required=False,
        choices=["FMoW","COOS"],
        default="COOS",
        help="CLIP model type",
    )

    args = parser.parse_args()
    
    model_name = args.model_name
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name)

    for split in ["val1","val2","val3","val4","train","test1","test2","test3","test4",]:
        dataloader = get_dataloader(args.dataset_name,split)
        embeddings = get_embeddings(dataloader, clip_processor, clip_model)
        # dict_embed = fix_embedding(embeddings)
        np.save("all_datasets/{}/embeddings/{}_embeddings.npy".format(args.dataset_name,split),embeddings)
        # np.save("all_datasets/embeddings/{}_{}_embeddings.npy".format(dataset_name,split),dict_embed)

if __name__ == "__main__":
    main()

