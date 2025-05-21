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
import json
import re
from task_dataset import TaskDataset


class ReIDDataset(TaskDataset):
    def __init__(self, split, subset_path=None, transform=None):
        super().__init__('ReID', split, subset_path)
        # apply transforms
        if transform is None:
            self.transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()])
        else: self.transform=transform
        # impute missing columns
        if "orientation" not in self.data.columns:
            self.data["orientation"] = ""
        # get individual indices
        self.data["individual_idx"] = pd.Categorical(self.data["dataset"].astype(str) + self.data["identity"].astype(str) + "___" + self.data["orientation"].astype(str)).codes
        self.num_classes = self.data["individual_idx"].nunique()
        # get encounter indices
        self.data["encounter_idx"] = pd.Categorical(self.data["image_id"]).codes


    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # getting raw data
        img_path = self.img_root_path+row['file_path']
        image = Image.open(img_path).convert("RGB")
        individual_idx = row["identity"]
        encounter_idx = row["encounter_idx"]
        uid = row['uid'] 
        species = row['object']

        # 
        text = f"image of {species} with identity {individual_idx}"
        label = individual_idx

        # cropping image
        if "bbox" in row and not isinstance(row["bbox"], float) and row["bbox"]!="[]":
            # print(row["bbox"])
            # bbox = re.sub(r"\s+", ",", row["bbox"]
            #         .replace("np.int64(", "")
            #         .replace("), ", ",")
            #         .replace(")", "")
            #         .replace(", ", ",")
            #     )
            bbox = list(map(float, re.findall(r'\d+', row["bbox"])))
            new_image = image.crop([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            if new_image.size[0] != 0 and new_image.size[1] !=0:
                image = new_image
        if self.transform:
            image = self.transform(image)
        print(label)
        return image, text, label, uid, encounter_idx

    def close(self):
        self.data.close()
 
if __name__ == '__main__':
    dataset=ReIDDataset('train')
    for i in range(len(dataset)):
        image, text, label, uid, encounter_idx=dataset[i]
        if i % 100 == 0:
            print(i)
    print("done")
    #print(image.size())







