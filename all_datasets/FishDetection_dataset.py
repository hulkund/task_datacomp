#GEODEDATASET
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import h5py
import io
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import transforms 
from task_dataset import TaskDataset

class FishDetectionDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None):
        super().__init__('FishDetection', split, subset_path)
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else: self.transform=transform

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        img_path = self.img_root_path+self.data.iloc[idx]['image_path']
        image = Image.open(img_path)
        text=None
        label=None
        uid = self.uids[idx]
        if self.transform:
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    image, text, label, uid=FishDetectionDataset('train')[1]
    print(text,label,uid)
    print(image)







