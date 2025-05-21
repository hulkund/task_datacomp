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
from task_dataset import TaskDataset


class iWildCamDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None):
        super().__init__('iWildCam', split, subset_path)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(256),  # Resize the shortest side to 256
                                             transforms.CenterCrop(224)])  # Center crop to 224x224
        self.data.dropna(subset=['labels'], inplace=True)
        self.data=self.data.reset_index()
        self.num_classes=len(np.unique(self.data['category_id']))
        self.total_samples=len(self.data)
        self.labels=self.data['labels']
        self.orig_labels=self.data['category_id']
        self.category_name=self.data['category_name']
        self.mapping=dict(zip(self.labels, self.category_name))
        self.img_path_col='filename'


    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        try:
            img_path = self.img_root_path+self.data.iloc[idx][self.img_path_col]
        except:
            print(idx)
            print(self.data.iloc[idx])
        image = Image.open(img_path)
        label = self.data.iloc[idx]['labels'] 
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
    image, text, label, uid=iWildCamDataset('test1')[1]
    print(image,text,label,uid)







