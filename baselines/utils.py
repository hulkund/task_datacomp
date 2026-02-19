import numpy as np
import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
from all_datasets.iWildCam_dataset import iWildCamDataset
# from all_datasets.iWildCamCropped_dataset import iWildCamCroppedDataset
from all_datasets.CivilComments_dataset import CivilCommentsDataset
from all_datasets.GeoDE_dataset import GeoDEDataset
from all_datasets.AutoArborist_dataset import AutoArboristDataset
from all_datasets.SelfDrivingCar_dataset import SelfDrivingCarDataset
from all_datasets.FishDetection_dataset import FishDetectionDataset
from all_datasets.ReID_dataset import ReIDDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import faiss
import numpy as np
import pdb
import yaml


def get_train_val_dl(dataset, batch_size, training_task):
    if training_task == "classification" :
        # Convert labels to a tensor (assuming dataset.labels is a pandas Series)
        all_labels = torch.tensor(dataset.labels.to_numpy())  
        print(len(all_labels), len(dataset))
        # Generate stratified train/val split indices
        train_indices, val_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.1,  # 10% validation set
            stratify=all_labels,  # Ensures class distribution is preserved
            random_state=42
        )
        train_labels = all_labels[train_indices]
        val_labels = all_labels[val_indices]
        # Compute number of unique classes
        num_classes = torch.unique(train_labels).numel()
    else:
        train_indices, val_indices = train_test_split(
            range(len(dataset)), 
            test_size=0.1,  # 10% validation set
            random_state=42
        )
        num_classes = dataset.num_classes
    # Create Subset objects for train and validation
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)  # No need to shuffle validation
    return train_dataset, val_dataset, train_dataloader, val_dataloader, num_classes

    

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

def get_threshold(embedding_path: str, key: str, fraction: float):
    """Compute a threshold from a fraction of top values in the embedding column.

    Returns:
        tuple: (threshold, embed_df)
    """
    embed_df = load_embedding(embedding_path, [key, "uid"])
    n = int(len(embed_df) * fraction)
    threshold = -np.sort(-embed_df[key].values)[n]
    return threshold, embed_df

def get_dataset(dataset_name,split,subset_path=None,transform=None,dataframe=None):
    if dataset_name == "iWildCam":
        dataset = iWildCamDataset(split=split, subset_path=subset_path, transform=transform, dataframe=dataframe)
    elif dataset_name == "GeoDE":
        dataset = GeoDEDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "AutoArborist":
        dataset = AutoArboristDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "SelfDrivingCar":
        dataset = SelfDrivingCarDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "FishDetection":
        dataset = FishDetectionDataset(split=split, subset_path=subset_path, transform=transform)
    elif dataset_name == "ReID":
        dataset = ReIDDataset(split=split, subset_path=subset_path, transform=transform)
    return dataset

def get_dataset_config(dataset_name):
    with open('../configs/datasets.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_metrics(predictions, ground_truth):
    """
    Compute overall and class-average accuracy metrics.

    Returns:
        dict: contains `acc` and `class_avg_acc`.
    """
    acc = accuracy_score(ground_truth, predictions)
    labels = np.unique(ground_truth)
    conf_mat = confusion_matrix(ground_truth, predictions, labels=labels)

    per_class_accs = {}
    row_sums = conf_mat.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.divide(
            conf_mat.diagonal(),
            row_sums,
            out=np.zeros_like(row_sums, dtype=float),
            where=row_sums != 0,
        )
    class_avg_acc = float(per_class_acc.mean()) if per_class_acc.size else 0.0
    metrics = {
        "accuracy": float(acc),
        "class_avg_accuracy": class_avg_acc,
    }
    return metrics

class FaissIndexIVFFlat:
    def __init__(self, data, nprobe=10):
        self.build(data, nprobe)

    def build(self, data, nprobe):
        nlist = int(np.sqrt(data.shape[0])) // 2
        quantizer = faiss.IndexFlatL2(data.shape[-1])
        self.index = faiss.IndexIVFFlat(quantizer, data.shape[-1], nlist)
        self.index.train(data)
        self.index.add(data)
        self.index.nprobe = nprobe

    def search(self, query, K):
        return self.index.search(query, K)
