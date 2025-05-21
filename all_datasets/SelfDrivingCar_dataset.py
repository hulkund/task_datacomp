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
import yaml
from task_dataset import TaskDataset


class NormalizeTo01(object):
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


class SelfDrivingCarDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None):
        super().__init__('SelfDrivingCar', split, subset_path)
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize(256),  # Resize the shortest side to 256 pixels
                                                transforms.CenterCrop(224),  # Center crop to 224x224
                                                NormalizeTo01(),])
                                            #transforms.ToTensor(),])
                                             # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             #                      std = [0.229, 0.224, 0.225])])
        else: self.transform=transform
        self.labels=self.data['steering']
        self.num_classes=1

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        img_path = self.img_root_path+self.data.iloc[idx]['filepath']
        image = Image.open(img_path)
        label = self.data.iloc[idx]['steering']
        uid = self.data.iloc[idx]['uid']
        location = self.data.iloc[idx]['location']
        speed = self.data.iloc[idx]['speed']
        text = "autonomous vehicle image of at speed {} with steering {}, taken at location".format(speed, label, location)
        if self.transform:
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    d=SelfDrivingCarDataset(split='train',subset_path='../experiments/SelfDrivingCar/random_filter_0.5/all_subset.npy')
    for i,t,l,u in d:
        print(u)







