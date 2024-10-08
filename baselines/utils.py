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
from all_datasets.CivilComments_dataset import CivilCommentsDataset
from all_datasets.GeoDE_dataset import GeoDEDataset
from all_datasets.AutoArborist_dataset import AutoArboristDataset
from all_datasets.SelfDrivingCar_dataset import SelfDrivingCarDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
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
    if dataset_name == "COOS":
        dataset = COOSDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "FMoW":
        dataset = FMoWDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "iWildCam":
        dataset = iWildCamDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "GeoDE":
        dataset = GeoDEDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "AutoArborist":
        dataset = AutoArboristDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "CropHarvest":
        dataset = CropHarvestDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "SelfDrivingCar":
        dataset = SelfDrivingCarDataset(split=split, subset_path=subset_path, transform=transform)
    return dataset

def get_metrics(predictions, ground_truth):
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro',labels=np.unique(ground_truth))
    recall = recall_score(ground_truth, predictions, average='macro',labels=np.unique(ground_truth))
    conf_mat = confusion_matrix(ground_truth, predictions,labels=np.unique(ground_truth))
    try:
        avg_acc = np.mean(conf_mat.diagonal()/conf_mat.sum(axis=1))
    except:
        return None
    metrics = {"acc":acc, "precision": precision, "recall":recall, "avg_acc":avg_acc}
    return metrics
