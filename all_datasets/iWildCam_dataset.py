#FMOW DATASET
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import h5py
import io
import numpy as np
from PIL import Image
import pandas as pd
import sys
from all_datasets.task_dataset import TaskDataset


class iWildCamDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None):
        super().__init__('iWildCam', split, subset_path)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(256),  # Resize the shortest side to 256
                                             transforms.CenterCrop(224)])  # Center crop to 224x224
        self.data.dropna(subset=['label'], inplace=True)
        self.data=self.data.reset_index()
        self.num_classes=len(np.unique(self.data['category_id']))
        self.labels=self.data['label']
        self.orig_labels=self.data['category_id']
        self.category_name=self.data['category_name']
        self.mapping=dict(zip(self.labels, self.category_name))
        self.img_path_col='filename'
        self.targets=torch.tensor(self.data['label'].to_numpy(), dtype=torch.long)
        self.classes=self.labels.unique()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            img_path = self.img_root_path+self.data.iloc[idx][self.img_path_col]
        except:
            print(idx)
            print(self.data.iloc[idx])
        image = Image.open(img_path)
        label = int(self.data.iloc[idx]['label'])
        label_str = self.data.iloc[idx]['category_name'] 
        uid = self.data.iloc[idx]['uid'] 
        time = self.data.iloc[idx]['date']
        text = "camera trap image of {}, taken on the date {}".format(label_str, time)
        if self.transform:
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    image, text, label, uid=iWildCamDataset('test4')[1]
    print(image,text,label,uid)







