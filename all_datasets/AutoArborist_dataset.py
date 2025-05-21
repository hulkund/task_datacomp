#CropHarvest DATASET
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

class NormalizeTo01(object):
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


class AutoArboristDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None):
        super().__init__('AutoArborist', split, subset_path)
        if transform is None:
            self.transform = transforms.Compose([NormalizeTo01()])
        else: self.transform=transform
        self.orig_labels=self.data['label']
        self.num_classes=len(np.unique(self.data['label']))
        self.labels=self.data['label']
        self.category_name=self.data['genus']
        self.mapping=dict(zip(self.labels, self.category_name))

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        img_path = self.img_root_path+self.data.iloc[idx]['street_level']
        image = Image.open(img_path)
        label = self.data.iloc[idx]['label'] 
        label_str = self.data.iloc[idx]['genus'] 
        uid = self.uids.iloc[idx]
        city = self.data.iloc[idx]['city']
        time = self.data.iloc[idx]['capturetime']
        text = "street-level Google Maps image of {} taken in city {}, USA, taken on the date {}".format(label_str, city, time)
        if self.transform:
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    image, text, label, uid=AutoArboristDataset('train')[1]
    print(image)
    print(text,label,uid)







