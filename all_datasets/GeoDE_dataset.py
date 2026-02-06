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
from all_datasets.task_dataset import TaskDataset

class GeoDEDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None, dataframe=None):
        super().__init__('GeoDE', split, subset_path, dataframe)
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        else: self.transform=transform
        self.num_classes=len(np.unique(self.data['label']))
        self.labels=self.data['label']
        self.targets = torch.tensor(self.data['label'].astype(int).to_numpy(), dtype=torch.long)
        self.category_name=self.data['object']
        self.mapping=dict(zip(self.labels, self.category_name))
        self.img_path_col='file_path'
        self.targets=torch.tensor(self.data['label'].to_numpy(), dtype=torch.long)
        self.classes=self.labels.unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_root_path+row[self.img_path_col]
        image = Image.open(img_path)
        label = row['label'] 
        label_str = row['object'] 
        uid = row['uid'] 
        country = row['ip_country']
        text = "image of {} taken in the country {}".format(label_str, country)
        if self.transform:
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    image, text, label, uid=GeoDEDataset('train')[1]
    print(image,text,label,uid)







