import sys
sys.path.append('/data/vision/beery/scratch/evelyn/task-datacomp/baselines/')

import multiprocessing as mp
import os
import time
from functools import partial
from multiprocessing import Pool
from queue import Empty
from typing import Any, List, Set, Tuple, Union

import fasttext
import fsspec
# import gcld3
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet
from tqdm import tqdm
import pdb
from baselines.utils import get_dataset


# def load_uids(embedding_path: str) -> np.ndarray:
#     """helper to read a embedding and load uids

#     Args:
#         fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

#     Returns:
#         np.ndarray: array of uids
#     """
#     df = load_embedding(embedding_path, columns=["uid"])
#     return np.array(df['uid'])

def get_common_labels(train_labels, test_labels):
    unique_train_labels = set(train_labels.unique())
    unique_test_labels = set(test_labels.unique())
    common_labels = unique_train_labels & unique_test_labels
    common_indices = train_labels[train_labels.isin(common_labels)].index
    print(f"unique val labels: {unique_test_labels}")
    print(f"{common_labels = }")
    return common_indices, common_labels

def match_label(dataset_name: str, task_name: int) -> np.ndarray:
    train_dataset = get_dataset(dataset_name=dataset_name,split="train")
    val_dataset = get_dataset(dataset_name=dataset_name,split=f"val{task_name}")
    train_labels = train_dataset.labels
    val_labels = val_dataset.labels
    train_uids = train_dataset.uids
    common_indices, _ = get_common_labels(train_labels, val_labels)
    return np.array(train_uids[common_indices])

def match_dist(dataset_name: str, task_name: int, fraction: float, random_seed: int = 42) -> np.ndarray:
    train_dataset = get_dataset(dataset_name=dataset_name,split="train")
    val_dataset = get_dataset(dataset_name=dataset_name,split=f"val{task_name}")
    print("got datasets")
    train_labels = train_dataset.labels
    val_labels = val_dataset.labels
    common_label_indices, common_labels=get_common_labels(train_labels, val_labels)
    desired_training_size = int(len(common_label_indices)*fraction)
    print("original desired training size:", int(len(train_labels)*fraction))
    print(f"{desired_training_size = }")
    
    train_labels = train_labels[common_label_indices]
    train_uids = train_dataset.uids[common_label_indices]
    min_instances_per_class=25

    print(f"{train_labels = }")
    
    rng = np.random.default_rng(seed=random_seed)

    train_distribution = train_labels.value_counts(normalize=True)
    val_distribution = val_labels.value_counts(normalize=True)
    sampled_uids_all = []
    for class_label in common_labels:
        class_size = len(train_labels[train_labels == class_label])
        print(f"{class_label = }, {class_size = }")
        if class_size > 0:
            target_size = max(min_instances_per_class, int(desired_training_size * val_distribution[class_label]))
            sampling_strategy = target_size / class_size
            sampled_indices = rng.choice(train_labels[train_labels == class_label].index, size=int(sampling_strategy * class_size), replace=True)
            sampled_uids = train_uids.loc[sampled_indices]
            sampled_uids_all.append(sampled_uids)
            print(f"selected {len(sampled_uids)} for cless {class_label}")
    final_train_uids = np.array(pd.concat(sampled_uids_all, ignore_index=True))
    return final_train_uids


# THEN AT THE END, THE MAIN FUNCTION
def apply_csv_filter(args: Any) -> None:
    """function to route the args to the proper baseline function

    Args:
        args (Any): commandline args

    Raises:
        ValueError: unsupported name
    """

    uids = None
    print(f"running: {args.name}")
    if args.name == "match_label":
        uids = match_label(
            dataset_name = args.dataset_name,
            task_name = args.task_name
        )
    elif args.name == "match_dist":
        if not args.random_seed:
            uids = match_dist(
                dataset_name = args.dataset_name, 
                task_name = args.task_name,
                fraction = args.fraction
            )
        else:
            uids = match_dist(
                dataset_name = args.dataset_name, 
                task_name = args.task_name,
                fraction = args.fraction,
                random_seed = args.random_seed
            )
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    print(f"sorting {len(uids)} uids")
    uids.sort()

    print(f"saving {args.save_path} with {len(uids)} entries")
    directory = os.path.dirname(args.save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(args.save_path, uids)
    
    

    

    

