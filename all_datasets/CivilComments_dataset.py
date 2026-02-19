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

root_path='/data/vision/beery/scratch/hasic/task-datacomp/all_datasets/CivilComments/civilcomments_v1.0/'
FILEPATHS = {'test1':'test1.csv',
             'test2':'test2.csv',
             'test3':'test3.csv',
             'test4':'test4.csv',
             'test5':'test5.csv',
             'test6':'test6.csv',
             'test7':'test7.csv',
             'train':'train.csv',
             'val1':'val1.csv',
             'val2':'val2.csv',
             'val3':'val3.csv',
             'val4':'val4.csv',
             'val5':'val5.csv',
             'val6':'val6.csv',
             'val7':'val7.csv'}

class CivilCommentsDataset(Dataset):
    def __init__(self, split, subset_path=None, transform=None):
        self.csv_path = root_path+FILEPATHS[split]
        self.data = pd.read_csv(self.csv_path)
        self.transform=transform
        if subset_path:
            uids_to_keep=np.load(subset_path,allow_pickle=True)
            self.data = self.data[self.data['uid'].isin(uids_to_keep)]
        # self.num_classes=len(np.unique(self.data['label']))
        self.total_samples = len(self.data)
        self.uids=self.data['uid']
        self.labels=(self.data['toxicity']>0.5).astype(int)
        

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        label = self.labels.iloc[idx]
        uid = self.data.iloc[idx]['uid'] 
        text = self.data.iloc[idx]['comment_text']
        if self.transform:
            text = self.transform(text)
        return text, text, label, uid

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    image, text, label, uid= CivilCommentsDataset('train')[1]
    print(CivilCommentsDataset('train').labels[1])
    print(image,text,label,uid)







