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

root_path='/data/vision/beery/scratch/neha/task-datacomp/all_datasets/AutoArborist/data'
FILEPATHS = {'test1':'test-senegal.csv',
             'test2':'test-tigray2020.csv',
             'test3':'test-tigray2021.csv',
             'train':'train.csv',
             'val1':'val-senegal.csv',
             'val2':'val-tigray2020.csv',
             'val3':'val-tigray2021.csv'}

# class NormalizeTo01(object):
#     def __call__(self, image):
#         if isinstance(image, Image.Image):
#             image = np.array(image)
#         image = image.astype(np.float32) / 255.0
#         return torch.from_numpy(image).permute(2, 0, 1)


class CropHarvestDataset(Dataset):
    def __init__(self, split, subset_path=None, transform=None):
        self.csv_path = root_path+FILEPATHS[split]
        self.data = pd.read_csv(self.csv_path)
        # if transform is None:
        #     self.transform = transforms.Compose([NormalizeTo01(),])
        # else: self.transform=transform
        if subset_path:
            uids_to_keep=np.load(subset_path,allow_pickle=True)
            self.data = self.data[self.data['uid'].isin(uids_to_keep)]
        self.num_classes=len(np.unique(self.data['label']))
        self.total_samples = len(self.data)
        self.uids=self.data['uid']
        self.labels=self.data['label']
        self.mapping=dict(zip(self.labels, self.category_name))

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        img_path = root_path+self.data.iloc[idx]['img_path']
        image = Image.open(img_path)
        label = self.data.iloc[idx]['label'] 
        label_str = self.data.iloc[idx]['category'] 
        uid = self.data.iloc[idx]['uid'] 
        region = self.data.iloc[idx]['region']
        time = self.data.iloc[idx]['timestamp'][:10]
        text = "satellite image of {} taken over {}, taken on the date {}".format(label_str, region, time)
        if self.transform:
            image = self.transform(image)
        return image, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    image, text, label, uid=CropHarvestDataset('train')[1]
    print(text,label,uid)







