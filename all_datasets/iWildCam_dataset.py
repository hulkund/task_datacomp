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

root_path='/data/vision/beery/scratch/neha/task-datacomp/all_datasets/iWildCam/'
FILEPATHS = {'test1':'test1.csv',
             'test2':'test2.csv',
             'test3':'test3.csv',
             'test4':'test4.csv',
             'train':'train.csv',
             'val1':'val1.csv',
             'val2':'val2.csv',
             'val3':'val3.csv',
             'val4':'val4.csv'}

# class RescaleTo01(object):
#     def __init__(self, min_value, max_value):
#         self.min_value = min_value
#         self.max_value = max_value

#     def __call__(self, sample):
#         sample = (sample - self.min_value) / (self.max_value - self.min_value)  # Rescale to [0, 1]
#         return sample

class NormalizeTo01(object):
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


class iWildCamDataset(Dataset):
    def __init__(self, split, subset_path=None, transform=None):
        self.csv_path = root_path+FILEPATHS[split]
        self.data = pd.read_csv(self.csv_path)
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize(256),  # Resize the shortest side to 256 pixels
                                                transforms.CenterCrop(224),  # Center crop to 224x224
                                                NormalizeTo01(),])
                                            #transforms.ToTensor(),])
                                             # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             #                      std = [0.229, 0.224, 0.225])])
        else: self.transform=transform
        if subset_path:
            uids_to_keep=np.load(subset_path,allow_pickle=True)
            self.data = self.data[self.data['uid'].isin(uids_to_keep)]
        self.num_classes=len(np.unique(self.data['category_id']))
        self.total_samples = len(self.data)
        self.uids=self.data['uid']
        self.labels=self.data['category_id']
        self.category_name=self.data['category_name']
        self.mapping=dict(zip(self.labels, self.category_name))


    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        img_path = root_path+'iwildcam_v2.0/train/'+self.data.iloc[idx]['filename']
        image = Image.open(img_path)
        label = self.data.iloc[idx]['category_id'] 
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
    image, text, label, uid=iWildCamDataset('train')[1]
    print(text,label,uid)







