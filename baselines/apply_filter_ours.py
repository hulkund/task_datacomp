import multiprocessing as mp
import os
import time
from functools import partial
from multiprocessing import Pool
from queue import Empty
from typing import Any, List, Set, Tuple, Union
import heapq

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
import sklearn
import yaml
import sys
from baselines.utils import FaissIndexIVFFlat

import argparse
from baselines.utils import get_dataset
from filters.caption_filter import caption_filter
from pathlib import Path

# import all the filters
from filters.gradmatch_filter import load_uids_with_gradmatch
from filters.image_align_filter import load_uids_with_image_alignment
from filters.basic_filter import load_uids_with_modality_filter
from filters.random_filter import load_uids_with_random_filter
from filters.clip_filter import load_uids_with_clip_score
from filters.text_alignment_filter import load_uids_with_text_alignment
from filters.tsds_filter import load_uids_with_tsds
from filters.utils import load_uids



def apply_filter(args: Any) -> None:
    """function to route the args to the proper baseline function

    Args:
        args (Any): commandline args

    Raises:
        ValueError: unsupported name
    """
    mp.set_start_method("spawn", force=True)

    uids = None
    print(f"running: {args.name}")
    print("args:", args)

    if args.name == "no_filter":
        uids = load_uids(
            args.embedding_path
        )
    # elif args.name == "basic_filter":
    #     uids = load_uids_with_basic_filter(
    #         args.embedding_path,
    #         args.num_workers,
    #     )
    elif args.name == "random_filter":
        uids = load_uids_with_random_filter(
            embedding_path=args.embedding_path,
            subset_percent=args.fraction
        )
    elif args.name == "image_based":
        uids = load_uids_with_modality_filter(
            val_embedding_path= args.val_embedding_path,
            pool_embedding_path=args.embedding_path,
            # val_centroids_path=args.centroids_path,
            pool_centroids_path=args.centroids_path,
            batch_size=16,
        )
    elif args.name == "text_based":
        uids = load_uids_with_modality_filter(
            val_embedding_path= args.val_embedding_path,
            pool_embedding_path=args.embedding_path,
            # val_centroids_path=args.centroids_path,
            pool_centroids_path=args.centroids_path,
            batch_size=16,
            key="text_embedding"
        )
    elif args.name == "clip_score":
        print(f"threshold {args.threshold} and fraction {args.fraction}")
        uids = load_uids_with_clip_score(
            embedding_path=args.embedding_path,
            threshold=args.threshold,
            fraction=args.fraction,
            num_workers=0,
        )
    elif args.name == "image_clip":
        print(f"threshold {args.threshold} and fraction {args.fraction}")
        uids = load_uids_with_modality_filter(
            val_embedding_path= args.val_embedding_path,
            pool_embedding_path=args.embedding_path,
            # val_centroids_path=args.centroids_path,
            pool_centroids_path=args.centroids_path,
            batch_size=args.batch_size,
            # arch=args.arch,
            threshold=args.threshold,
            fraction=args.fraction,
            # num_workers=args.num_workers,
        )
    elif args.name == "image_alignment":
        uids = load_uids_with_image_alignment(
            val_embedding_path= args.val_embedding_path,
            pool_embedding_path=args.embedding_path,
            fraction=args.fraction
        )
    elif args.name == "text_alignment":
        uids = load_uids_with_text_alignment(
            val_embedding_path= args.val_embedding_path,
            pool_embedding_path=args.embedding_path,
            fraction=args.fraction
        )
    elif args.name == "tsds":
        uids = load_uids_with_tsds(
            val_embedding_path=args.val_embedding_path,
            pool_embedding_path=args.embedding_path
        )
    elif args.name == "gradmatch":
        args.model = 'ResNet18'
        uids = load_uids_with_gradmatch(
            fraction=args.fraction, 
            balance=True, 
            lam=1.0,
            args=args,
        )
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    print(f"sorting {len(uids)} uids")
    uids.sort()

    print(f"saving {args.save_path} with {len(uids)} entries")
    print("uids:", uids)

    directory = os.path.dirname(args.save_path)
    print("os.pwd", os.getcwd())
    print("this is the directory that needs to be created", directory)
    if not os.path.exists(directory):
        print("creating directory")
        os.makedirs(directory)
    print("saving...")
    np.save(args.save_path, uids)
    print("saved")
    print(f"File size: {os.path.getsize(args.save_path)} bytes")
    
    

    

    

