import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import h5py
import io
import numpy as np
import yaml
from PIL import Image
import pandas as pd
import sys
sys.path.append("../")
sys.path.append("/data/vision/beery/scratch/neha/task-datacomp/")



with open('configs/datasets.yaml', 'r') as file:
    data = yaml.safe_load(file)

class TaskDataset(Dataset):
    def __init__(self, dataset_name, split, subset_path, dataframe=None):
        if dataframe is not None:
            self.data = dataframe
        else:
            self.csv_path = data[dataset_name]['csv_root_path']+data[dataset_name]['FILEPATHS'][split]
            print(self.csv_path)
            self.data = pd.read_csv(self.csv_path)
        self.img_root_path = data[dataset_name]['img_root_path']
        if subset_path:
            uids_to_keep=np.load(subset_path,allow_pickle=True)
            print(f"{len(uids_to_keep) = }")
            print(f"{sorted(uids_to_keep)[0:10] = }")
            missing_uids = set(uids_to_keep) - set(self.data['uid'])
            print("Missing UIDs:", missing_uids)
            self.data = self.data[self.data['uid'].isin(uids_to_keep)]
            print(f"{len(self.data['label']) = }")
        self.total_samples = len(self.data)
        self.uids=self.data['uid']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        raise NotImplementedError