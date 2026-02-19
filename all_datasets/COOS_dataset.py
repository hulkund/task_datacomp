import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import h5py
import io
import numpy as np
from PIL import Image
import pandas as pd

root_path='/data/vision/beery/scratch/hasic/task-datacomp/all_datasets/COOS/data/'
FILEPATHS = {'test1':'test1_data.h5',
             'test2':'test2_data.h5',
             'test3':'test3_data.h5',
             'test4':'test4_data.h5',
             'train':'train_data.h5',
             'val1':'val1_data.h5',
             'val2':'val2_data.h5',
             'val3':'val3_data.h5',
             'val4':'val4_data.h5'}

class RescaleTo01(object):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sample):
        sample = (sample - self.min_value) / (self.max_value - self.min_value)  # Rescale to [0, 1]
        sample = torch.cat([sample,sample,sample], dim=0)
        return sample

class COOSDataset(Dataset):
    def __init__(self, split, subset_path=None, transform=None):
        self.file_path = root_path+FILEPATHS[split]
        self.data_file = h5py.File(self.file_path, 'r')
        self.uids=pd.DataFrame(self.data_file['uid'])[0]
        self.labels=pd.DataFrame(self.data_file['labels'])[0]
        self.images=self.data_file['data']
        
        if subset_path:
            uids_to_keep=np.load(subset_path,allow_pickle=True)
            id_indices = np.isin(uids_to_keep, self.uids).nonzero()[0]
            self.labels = self.labels[id_indices]
            self.images = self.images[id_indices]
            self.uids = self.uids[id_indices]
            
        self.total_samples = len(self.images)
        self.min = np.min(self.images[:])
        self.max = np.max(self.images[:])
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                             RescaleTo01(self.min,self.max),])
            
        else: self.transform=transform
        self.num_classes=7
        self.mapping = {0:'Endoplasmic Reticulum',
                        1:'Inner Mitochondrial Membrane',
                        2:'Golgi',
                        3:'Peroxisomes',
                        4:'Early Endosome',
                        5:'Cytosol',
                        6:'Nuclear Envelope'}
        self.category_name = [self.mapping[label] for label in self.labels]
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        image = np.expand_dims(self.images[idx][0,:,:],axis=2)
        label = self.labels[idx]  # Assuming labels are stored in 'labels' dataset
        uid = self.uids[idx]  # Assuming labels are stored in 'labels' dataset
        text = "microscopy image of mouse cell cropped around the {}".format(self.mapping[label])
        if self.transform:
            image=Image.fromarray(image.squeeze(2))
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data_file.close()




