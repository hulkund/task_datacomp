import numpy as np
import os
import clip
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
from all_datasets.FMoW_dataset import FMoWDataset
from all_datasets.COOS_dataset import COOSDataset
from all_datasets.iWildCam_dataset import iWildCamDataset
import pandas as pd

def fix_embedding(embed):
    embedding = {}
    for e in embed:
        for key, value in e.items():
            if key not in embedding:
                embedding[key] = []
            embedding[key].append(value)
    return embedding

def load_embedding(embedding_path:str, columns):
    embed = np.load(f"{embedding_path}",allow_pickle=True)
    embed_df=pd.DataFrame()
    for col in columns:
        embed_df[col] = [e[col] for e in embed]
    return embed_df

def get_dataset(dataset_name,split,subset_path=None,transform=None):
    print(dataset_name)
    if dataset_name == "COOS":
        dataset = COOSDataset(split=split,subset_path=subset_path,transform=transform)
    elif dataset_name == "FMoW":
        dataset = FMoWDataset(split=split,subset_path=subset_path,transform=transform)
    elif dataset_name == "iWildCam":
        dataset = iWildCamDataset(split=split,subset_path=subset_path,transform=transform)
    return dataset