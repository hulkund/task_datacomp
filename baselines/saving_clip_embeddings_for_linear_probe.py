import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')
from utils import get_dataset
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json
from all_datasets.FMoW_dataset import FMoWDataset
from all_datasets.COOS_dataset import COOSDataset
import torch
from all_datasets.iWildCam_dataset import iWildCamDataset
import clip
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_features(model,dataset):
    all_features = []
    all_labels = []
    all_uids = []
    with torch.no_grad():
        for images,_, labels, uids in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
            all_uids.append(list(uids))
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), np.concatenate(all_uids)

def get_text_features(model,dataset):
    all_features = []
    all_labels = []
    all_uids = []
    with torch.no_grad():
        for _, texts, labels, uids in tqdm(DataLoader(dataset, batch_size=100)):
            texts=clip.tokenize(texts).to(device)
            features = model.encode_text(texts.to(device))
            all_features.append(features)
            all_labels.append(labels)
            all_uids.append(list(uids))
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy(), np.concatenate(all_uids)

def get_model_processor():
    model, preprocess = clip.load('ViT-B/32', device)
    return model, preprocess

parser = argparse.ArgumentParser(description="")
parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["FMoW","COOS","iWildCam"],
        default="COOS",
        help="Dataset name",
    )
parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="split",
)
args = parser.parse_args()


model, preprocess = get_model_processor()
print("got model processor")

dataset = get_dataset(dataset_name=args.dataset_name,
                             split=args.split,
                            subset_path=None,
                            transform=preprocess)
if dataset == 'CivilComments':
    continue
else:
    print("got dataset")
    features, labels, uids = get_image_features(model, dataset)


print("got features")
save_path=f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{args.dataset_name}/{args.split}_image_label_embed.npz"
np.savez(save_path, labels=labels, features=features, uids=uids)





    