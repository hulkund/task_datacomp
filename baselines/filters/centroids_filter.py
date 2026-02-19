import torch
from typing import Union
from baselines.utils import load_embedding
import numpy as np
import pandas as pd

@torch.no_grad()
def get_centroid_ids_gpu(
    features: torch.Tensor, centroids: torch.Tensor, batch_size: int, device=0
) -> torch.Tensor:
    """assign features to closest centroid

    Args:
        features (torch.Tensor): features to assign to centroids
        centroids (torch.Tensor): reference centroids
        batch_size (int): gpu batch size
        device (int): gpu number

    Returns:
        torch.Tensor: assignment of features to labels
    """
    device_string = f"cuda:{device}"
    centroids_gpu = centroids.to(device_string)
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    for i in range(0, features.shape[0], batch_size):
        similarity = torch.einsum(
            "ik, jk -> ij",
            features[i : i + batch_size, :].float().to(device_string),
            centroids_gpu,
        )
        matches = torch.argmax(similarity, dim=1).cpu()
        labels[i : i + batch_size] = matches.long()
    return labels

def load_uids_with_modality_filter(
    val_embedding_path: str,
    pool_embedding_path: str,
    # val_centroids_path: str,
    pool_centroids_path: str,
    batch_size:int,
    threshold: Union[float,None]=None,
    fraction:Union[float, None] = None,
    key:str="image_embedding"
) -> np.ndarray:
    
    sim_key = "similarity_score"
    val_embed_df=load_embedding(val_embedding_path, [key,"uid"])
    pool_embed_df=load_embedding(pool_embedding_path, [key,"uid"])
    val_embedding = torch.Tensor(val_embed_df[key])
    pool_embedding = torch.Tensor(pool_embed_df[key])

    pool_centroids = torch.from_numpy(
        torch.load(pool_centroids_path)
    )
    target_centroid_ids = get_centroid_ids_gpu(
        features=val_embedding, 
        centroids=pool_centroids, 
        batch_size=batch_size
        )
    target_centroid_ids = torch.unique(target_centroid_ids)
    uids=pool_embed_df.uid
    print(len(uids))
    candidate_centroid_ids = get_centroid_ids_gpu(
            features=pool_embedding,#[mask],
            centroids=pool_centroids,
            batch_size=batch_size,
        )
    centroid_id_to_uids = {}
    for uid, label in zip(uids, candidate_centroid_ids):
        centroid_id_to_uids.setdefault(label.item(), []).append(uid)
    uids_to_keep = []
    for i in target_centroid_ids:
        if i.item() in centroid_id_to_uids:
            uids_to_keep.extend(centroid_id_to_uids[i.item()])
    # print(len(uids_to_keep))
    uids_to_keep = np.unique(uids_to_keep)
    print(len(uids_to_keep))
    return np.array(uids_to_keep)