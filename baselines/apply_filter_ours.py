import time

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
from DeepCore.deepcore.methods.gradmatch import GradMatch
from DeepCore.deepcore.methods.glister import Glister

from torch.utils.data import Subset

from pathlib import Path

import yaml

with open("configs/datasets.yaml", "r") as f:
    DATASETS = yaml.safe_load(f)

import zcore.core.coreset as cs

def get_fasttext_language(text: str, lang_detect_model: Any) -> str:
    """helper to detect language of a piece of text (fasttext)

    Args:
        text (str): text whose language we want to determing
        lang_detect_model (Any): fasttext model to detect langauge

    Returns:
        str: ISO language code
    """
    text = text.replace("\n", " ")
    language = lang_detect_model.predict(text)[0][0].split("__label__")[1]
    return language

def caption_filter(df: pd.DataFrame, lang_detect_model: Any) -> np.ndarray:
    """apply a low-level text filter for the image based baseline

    Args:
        df (pd.DataFrame): parquet metadata
        lang_detect_model (Any): fasttext model

    Returns:
        np.ndarray: boolean numpy array containing selected entries
    """
    caption_num_words = df.text.apply(lambda x: len(fasttext.tokenize(x)))
    caption_num_chars = df.text.apply(len)

    lang_preds, _ = lang_detect_model.predict(
        [x.replace("\n", " ") for x in df.text.values], k=1
    )
    fasttext_en = [x[0].replace("__label__", "") == "en" for x in lang_preds]

    mask = fasttext_en & (caption_num_words > 1) & (caption_num_chars > 5)

    return mask.to_numpy()


def load_uids_with_basic_filter_helper(embedding_path) -> np.ndarray:
    """helper to run basic filter on a single parquet

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    # make this uid, text, original width, original height
    df = torch.from_numpy(np.load(embedding_path))#['image_embedding'])

    lang_detect_model = fasttext.load_model(download("fasttext", "~/.cache/fasttext"))

    fasttext_lang_pred = df.text.apply(
        lambda x: get_fasttext_language(x, lang_detect_model)
    )
    caption_num_words = df.text.apply(lambda x: len(x.split()))
    caption_num_chars = df.text.apply(lambda x: len(x))
    uid_int = df.uid.apply(int, base=16)
    uid_upper_uint64 = (uid_int // 2**64).astype("uint64")
    uid_lower_uint64 = (uid_int % 2**64).astype("uint64")

    inds_array = np.array(list(zip(uid_upper_uint64, uid_lower_uint64)), "u8,u8")

    english_mask = fasttext_lang_pred == "en"
    caption_mask = (caption_num_words > 2) & (caption_num_chars > 5)
    min_image_dim = np.minimum(df.original_width, df.original_height)
    max_image_dim = np.maximum(df.original_width, df.original_height)
    aspect_ratio = max_image_dim / min_image_dim
    image_mask = (min_image_dim >= 200) & (aspect_ratio <= 3.0)

    return inds_array[english_mask & caption_mask & image_mask]
    
def load_uids_with_clip_score(
    embedding_path: str,
    threshold: float,
    fraction: float,
    num_workers: int,
    train_csv_path = None,
    val_csv_path = None,
) -> np.ndarray:
    """load in metadata with a threshold applied

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        threshold (float): threshold value to apply on the key column
        fraction (float): fraction value to apply on the key column
        num_workers (int): number of cpu workers, each of which processes a parquet
        gcld3_en_filter (bool): if True, apply gcld3 english filtering (used for laion2b filter)
                                Default False.

    Returns:
        np.ndarray: array of uids
    """

    # NOTE: assumes errors already checked as in baselines.py
    key = "similarity_score"
    if threshold is None:
        # convert a fraction into a threshold
        threshold, embed_df = get_threshold(
            embedding_path,
            key,
            fraction,
            train_csv_path,
            val_csv_path
        )
    else: 
        embed_df = load_embedding(
            embedding_path,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path
        )
    uids=np.array([uid for uid in embed_df[embed_df[key] >= threshold]["uid"]])
    return uids

def load_uids_with_image_alignment(
    pool_embedding_path: str,
    val_embedding_path: str,
    fraction: float,
) -> np.ndarray:
    """load in metadata with a threshold applied

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        threshold (float): threshold value to apply on the key column
        fraction (float): fraction value to apply on the key column
        num_workers (int): number of cpu workers, each of which processes a parquet
        gcld3_en_filter (bool): if True, apply gcld3 english filtering (used for laion2b filter)
                                Default False.

    Returns:
        np.ndarray: array of uids
    """

    # NOTE: assumes errors already checked as in baselines.py
    key = "image_embedding"
    pool_embed = load_embedding(pool_embedding_path, [key, "uid"])
    val_embed = load_embedding(val_embedding_path, [key, "uid"])
    batch_size=100
    # set k to be 10 then 20, etc
    similarities=[]
    for i in range(0, len(pool_embed), batch_size):
        pool_embed_batch = np.vstack(pool_embed[key][i:i+batch_size])
        val_embed_batch = np.vstack(val_embed[key])
        similarity = sklearn.metrics.pairwise.cosine_similarity(pool_embed_batch, val_embed_batch)
        # needs to be changed 
        similarity_per_sample = np.sum(similarity, axis=1)
        similarities.append(similarity_per_sample)
    key_similarity = 'similarity_to_val'
    pool_embed[key_similarity] = np.hstack(similarities)
    # get threshold
    n = int(len(pool_embed) * fraction)
    threshold = -np.sort(-pool_embed[key_similarity].values)[n]
    uids=np.array([uid for uid in pool_embed[pool_embed[key_similarity] >= threshold]["uid"]])
    return uids

def load_uids_with_text_alignment(
    pool_embedding_path: str,
    val_embedding_path: str,
    fraction: float,
) -> np.ndarray:
    """load in metadata with a threshold applied

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        threshold (float): threshold value to apply on the key column
        fraction (float): fraction value to apply on the key column
        num_workers (int): number of cpu workers, each of which processes a parquet
        gcld3_en_filter (bool): if True, apply gcld3 english filtering (used for laion2b filter)
                                Default False.

    Returns:
        np.ndarray: array of uids
    """

    # NOTE: assumes errors already checked as in baselines.py
    key = "text_embedding"
    pool_embed = load_embedding(pool_embedding_path, [key, "uid"])
    val_embed = load_embedding(val_embedding_path, [key, "uid"])
    batch_size=1000
    similarities=[]
    for i in range(0, len(pool_embed), batch_size):
        pool_embed_batch = np.vstack(pool_embed[key][i:i+batch_size])
        val_embed_batch = np.vstack(val_embed[key])
        similarity = sklearn.metrics.pairwise.cosine_similarity(pool_embed_batch, val_embed_batch)
        similarity_per_sample = np.sum(similarity, axis=1)
        similarities.append(similarity_per_sample)
    key_similarity = 'similarity_to_val'
    pool_embed[key_similarity] = np.hstack(similarities)
    # get threshold
    n = int(len(pool_embed) * fraction)
    threshold = -np.sort(-pool_embed[key_similarity].values)[n]
    uids=np.array([uid for uid in pool_embed[pool_embed[key_similarity] >= threshold]["uid"]])
    return uids

# def load_uids_with_cluster_ot_filter(
#     pool_embedding_path: str,
#     val_embedding_path: str,
#     fraction: float,
# ) -> np.ndarray:
#     key = "image_embedding"
#     pool_embed = load_embedding(pool_embedding_path, [key, "uid"])
#     val_embed = load_embedding(val_embedding_path, [key, "uid"])

    


def load_embedding(embedding_path: str, columns, train_csv_path = None, val_csv_path = None):
    embed = np.load(f"{embedding_path}",allow_pickle=True)
    embed_df=pd.DataFrame()
    for col in columns:
        embed_df[col] = [e[col] for e in embed]
    
    if train_csv_path and val_csv_path:
        print("in here")
        train_csv_df = pd.read_csv(train_csv_path)
        uid_to_label_map = {row["uid"] : row["label"] for _, row in train_csv_df.iterrows()}

        val_csv_df = pd.read_csv(val_csv_path)
        print(f"{len(val_csv_df)=}")
        val_labels = set(val_csv_df["label"])
        print("val_labels:", val_labels)

        allowed_uids = {uid for uid, label in uid_to_label_map.items() if label in val_labels}
        print("len(allowed_uids):", len(allowed_uids))

        embed_df = embed_df[embed_df["uid"].isin(allowed_uids)]
        embed_df = embed_df.reset_index(drop=True)
  
    print("len(embed_df):", len(embed_df))
    return embed_df

def get_threshold(
    embedding_path: str, key: str, fraction: float, train_csv_path = None, val_csv_path = None,
) -> float:
    """compute a threshold given a collection of metadata, a key, and a target fraction of the pool to keep

    Args:
        metadata_dir_path (str): directory where metadata is stored
        key (str): column we are interested in the parquet column store
        fraction (float): top k fraction, represented as a decimal.
        num_workers (int): number of cpu workers, each of which processes a parquet.

    Returns:
        float: threshold value
    """
    print("loading all metadata for threshold computation")
    embed_df = load_embedding(
        embedding_path,
        columns=[key,"uid"],
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path
    )
    n = int(len(embed_df) * fraction)
    threshold = -np.sort(-embed_df[key].values)[n]
    return threshold, embed_df

@torch.no_grad()
def get_centroid_ids_gpu(
    features: torch.Tensor, centroids: torch.Tensor, batch_size: int, device=0
) -> torch.Tensor:
    """assign features to closest centroid

    Args:
        features (torch.Tensor): features to assign to centroids
        centroids (torch.Tensor): reference centroids
        batch_size (int): gpu batch size
        device (int): gpu number

    Returns:
        torch.Tensor: assignment of features to labels
    """
    device_string = f"cuda:{device}"
    centroids_gpu = centroids.to(device_string)
    labels = torch.zeros(features.shape[0], dtype=torch.long)
    for i in range(0, features.shape[0], batch_size):
        similarity = torch.einsum(
            "ik, jk -> ij",
            features[i : i + batch_size, :].float().to(device_string),
            centroids_gpu,
        )
        matches = torch.argmax(similarity, dim=1).cpu()
        labels[i : i + batch_size] = matches.long()
    return labels

def load_uids_with_modality_filter(
    val_embedding_path: str,
    pool_embedding_path: str,
    # val_centroids_path: str,
    pool_centroids_path: str,
    batch_size:int,
    threshold: Union[float,None]=None,
    fraction:Union[float, None] = None,
    key:str="image_embedding"
) -> np.ndarray:
    
    sim_key = "similarity_score"
    val_embed_df=load_embedding(val_embedding_path, [key,"uid"])
    pool_embed_df=load_embedding(pool_embedding_path, [key,"uid"])
    val_embedding = torch.Tensor(val_embed_df[key])
    pool_embedding = torch.Tensor(pool_embed_df[key])

    pool_centroids = torch.from_numpy(
        torch.load(pool_centroids_path)
    )
    target_centroid_ids = get_centroid_ids_gpu(
        features=val_embedding, 
        centroids=pool_centroids, 
        batch_size=batch_size
        )
    target_centroid_ids = torch.unique(target_centroid_ids)
    uids=pool_embed_df.uid
    print(len(uids))
    candidate_centroid_ids = get_centroid_ids_gpu(
            features=pool_embedding,#[mask],
            centroids=pool_centroids,
            batch_size=batch_size,
        )
    centroid_id_to_uids = {}
    for uid, label in zip(uids, candidate_centroid_ids):
        centroid_id_to_uids.setdefault(label.item(), []).append(uid)
    uids_to_keep = []
    for i in target_centroid_ids:
        if i.item() in centroid_id_to_uids:
            uids_to_keep.extend(centroid_id_to_uids[i.item()])
    # print(len(uids_to_keep))
    uids_to_keep = np.unique(uids_to_keep)
    print(len(uids_to_keep))
    return np.array(uids_to_keep)


def load_uids(embedding_path: str, train_csv_path = None, val_csv_path = None) -> np.ndarray:
    """helper to read a embedding and load uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    df = load_embedding(
        embedding_path,
        columns=["uid"],
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path
    )
    print("return length:", len(np.array(df['uid'])))
    return np.array(df['uid'])

def load_uids_with_random_filter(
    embedding_path: str,
    subset_percent: float,
    random_seed: int,
    train_csv_path = None,
    val_csv_path = None
) -> np.ndarray:
    embed_df=load_embedding(
        embedding_path,
        columns=["uid"],
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path
    )
    uids_selected=embed_df.sample(frac=subset_percent,random_state=random_seed)
    return np.array(uids_selected).squeeze(axis=1)

def load_uids_with_tsds(
    val_embedding_path: str, 
    pool_embedding_path: str,
    fraction: int,
    random_seed: int,
    train_csv_path = None,
    val_csv_path = None
) -> np.ndarray:

    key = "image_embedding"
    query_df=load_embedding(val_embedding_path, [key,"uid"])
    candidate_df=load_embedding(pool_embedding_path, [key,"uid"], train_csv_path=train_csv_path, val_csv_path=val_csv_path)
    # candidate_df=load_embedding(pool_embedding_path, [key,"uid"])
    
    xq=np.stack(query_df["image_embedding"])
    xb=np.stack(candidate_df["image_embedding"])

    with open("baselines/tsds_data/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    SAMPLE_SIZE = int(len(candidate_df) * fraction)
    print(f"{SAMPLE_SIZE = }")

    MAX_K = config["max_K"]
    # KDE_K = config["kde_K"]
    KDE_K = int(len(candidate_df) ** 0.5)
    print(f"{KDE_K = }")
    SIGMA = config["sigma"]
    ALPHA = config["alpha"]
    C = config["C"]

    MAX_K = min(MAX_K, xb.shape[0] // 10)
    KDE_K = min(KDE_K, xb.shape[0] // 10)
    
    # logging.info(f"Starting building index for the candidate examples.")
    index = FaissIndexIVFFlat(xb)
    
    # logging.info(f"Start prefetching {MAX_K}-nearest neighbors for each query example.")
    top_dists, top_indices = index.search(xq, MAX_K)
    top_indices = top_indices.astype(int)
    sorted_indices = np.argsort(top_dists, axis=-1)
    static_indices = np.indices(top_dists.shape)[0]
    top_dists = np.sqrt(top_dists[static_indices, sorted_indices])
    # top_indices[i][j] is the index of the jth nearest neighbor
    # (among the candidates) of the ith query example
    top_indices = top_indices[static_indices, sorted_indices]
    
    # top_kde[i][j] is the KDE of the jth nearest neighbor of the ith query example
    if SIGMA == 0:
        # logging.info("Sigma is zero, KDE (kernel density estimation) set to 1 for all the points.")
        top_kdes = np.ones_like(top_indices)
    else:
        # logging.info(f"Start computing KDE (kernel density estimation), neighborhood size: {KDE_K}.")
        top_indices_set = list(set([i for i in top_indices.reshape(-1)]))
        top_features = xb[top_indices_set]
        index_for_kde = FaissIndexIVFFlat(top_features)
        D2, I = index_for_kde.search(top_features, KDE_K)
        kernel = 1 - D2 / (SIGMA ** 2)
        print(f'A point has {(kernel > 0).sum(axis=-1).mean() - 1} near-duplicates on average.')
        kernel = kernel * (kernel > 0)
        kde = kernel.sum(axis=-1)
        kde_map = {top_indices_set[i]:kde[i] for i in range(len(top_indices_set))}
        kde_mapfunc = np.vectorize(lambda t: kde_map[t])
        top_kdes = kde_mapfunc(top_indices)
    
    print(f"{top_kdes.size = }")
    nan_count = np.isnan(top_kdes).sum()
    print(f"top_kdes {nan_count=}")
            
    # logging.info("Start computing the probability assignment.")
    M, N = top_indices.shape[0], xb.shape[0]
    lastK = [0] * M
    heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(M)]
    heapq.heapify(heap)
    dist_weighted_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(M)]
    s = 0
    cost = np.zeros(M)
    total_cost = 0
    while len(heap) > 0:
        count, curr_k, curr_j = heapq.heappop(heap)
        s = count
        # if we increase s by any positive amount, the 0, 1, ..., curr_k has to transport probability mass to curr_k + 1
        total_cost -= cost[curr_j]
        cost[curr_j] = top_dists[curr_j][curr_k + 1] * count - dist_weighted_sum[curr_j]
        total_cost += cost[curr_j]
        # If the condition breaks, the current s will be the final s
        if ALPHA / C * total_cost >= (1 - ALPHA) * M:
            break
        lastK[curr_j] = curr_k
        if curr_k < MAX_K - 2:
            count += 1.0 / top_kdes[curr_j][curr_k + 1]
            heapq.heappush(heap, (count, curr_k + 1, curr_j))
            dist_weighted_sum[curr_j] += top_dists[curr_j][curr_k + 1] / top_kdes[curr_j][curr_k + 1]
    global_probs = np.zeros(N)
    for j in range(M):
        prob_sum = 0
        for k in range(lastK[j] + 1):
            global_probs[top_indices[j][k]] += 1 / M / s / top_kdes[j][k]
            prob_sum += 1 / M / s / top_kdes[j][k]
        global_probs[top_indices[j][lastK[j] + 1]] += max(1.0 / M - prob_sum, 0)
        assert 1.0 / M - prob_sum >= -1e-9, f'{1.0 / M - prob_sum}'
        assert (1.0 / M - prob_sum) * top_kdes[j][lastK[j] + 1] * M * s <= 1 + 1e-9 or lastK[j] == MAX_K - 2, f'{(1.0 / M - prob_sum) * top_kdes[j][lastK[j] + 1] * M * s}'
    
    # logging.info(f"Start sampling. Sample size: {SAMPLE_SIZE}.")
    rng = np.random.default_rng(seed=random_seed)
    sample_times = rng.multinomial(SAMPLE_SIZE, global_probs)
    sample_indices = []
    for i in range(sample_times.shape[0]):
        sample_indices.extend([i] * sample_times[i])

    uid_array = candidate_df["uid"].to_numpy()
    selected_uids = np.array([uid_array[i] for i in sample_indices])
    return selected_uids

def gradmatch_acf_mapping(train_df, val_df, fraction):
    acf_mapping = {}
    print(f"{len(train_df) = }")
    print(f"{len(val_df) = }")

    val_classes = val_df["label"].unique()
    for c in val_classes:
        val_class_frac = (val_df["label"] == c).mean()
        train_class_frac = (train_df["label"] == c).mean()
        print(f"class={c}: len(train_class)={(train_df['label'] == c).sum()}, len(val_class)={(val_df['label'] == c).sum()}")
        if train_class_frac == 0:
            selection_class_frac = 0
        else:
            selection_class_frac = min(1, fraction * val_class_frac / train_class_frac)
        acf_mapping[c] = selection_class_frac
    print("acf_mapping:", acf_mapping)
    return acf_mapping

def _prepare_deepcore_method_inputs(args):
    """
    Shared setup for GradMatch and Glister: build args, filter train set to keep
    only the labels shared in the query set.
    """
    
    args_dict = {
        'print_freq': 100,
        'device': 'cuda',
        'workers': 4,
        'model': args.model,
        'selection_optimizer': 'SGD',
        'selection_momentum': 0.9,
        'selection_weight_decay': 1e-4,
        'selection_nesterov': True,
        'selection_test_interval': 10,
        'selection_test_fraction': 1.0,
        'specific_model': None,
        'torchvision_pretrain': True,
        'if_dst_pretrain': False,
        'dst_pretrain_dict': {},
        'n_pretrain_size': 1000,
        'n_pretrain': 1000,
        'gpu': [0]
    }

    p = Path(args.val_embedding_path)
    filename = p.name
    dataset_name = p.parent.parent.name
    val_split = filename.split('_')[0]
    print(f"dataset={dataset_name}, val_split={val_split}")

    train_dataset = get_dataset(dataset_name, split='train')
    val_dataset = get_dataset(dataset_name, split=val_split)

    print(f"Before filtering: train={len(train_dataset.data)}, val={len(val_dataset.data)}")

    val_labels = set(np.unique(val_dataset.labels))
    train_labels = set(np.unique(train_dataset.labels))
    common_labels = val_labels.intersection(train_labels)

    train_indices = train_dataset.labels[train_dataset.labels.isin(common_labels)].index
    val_indices = val_dataset.labels[val_dataset.labels.isin(common_labels)].index

    label_map = {old: new for new, old in enumerate(sorted(common_labels))}

    filtered_train_df = train_dataset.data.iloc[train_indices].reset_index(drop=True)
    filtered_train_df["label"] = filtered_train_df["label"].map(label_map)

    filtered_val_df = val_dataset.data.iloc[val_indices].reset_index(drop=True)
    filtered_val_df["label"] = filtered_val_df["label"].map(label_map)

    train_dataset = get_dataset(dataset_name, split='train', dataframe=filtered_train_df)
    val_dataset = get_dataset(dataset_name, split=val_split, dataframe=filtered_val_df)

    classes_match = len(train_dataset.classes) == len(val_dataset.classes)
    print(f"After filtering: train={len(train_dataset.data)}, val={len(val_dataset.data)}, classes_match={classes_match}")

    args_dict['num_classes'] = train_dataset.data['label'].nunique()
    args_dict['channel'] = 3
    args_dict['im_size'] = [224, 224]

    user_args = vars(args)
    if 'selection_lr' in user_args:
        user_args['selection_lr'] = float(user_args['selection_lr'])
    if 'selection_batch' in user_args:
        user_args['selection_batch'] = int(user_args['selection_batch'])
    if 'lam' in user_args:
        user_args['lam'] = float(user_args['lam'])

    merged_dict = {**user_args, **args_dict}
    merged_args = argparse.Namespace(**merged_dict)

    return train_dataset, val_dataset, merged_args

def load_uids_with_gradmatch(
    fraction=0.25,
    random_seed=42,
    epochs=50,
    balance=True,
    args=None
) -> np.ndarray:
    train_dataset, val_dataset, merged_args = _prepare_deepcore_method_inputs(args)

    acf_mapping = None
    if args.name == "gradmatch_acf":
        acf_mapping = gradmatch_acf_mapping(train_dataset.data, val_dataset.data, fraction)

    gradmatch = GradMatch(
        dst_train=train_dataset,
        fraction=fraction,
        random_seed=random_seed,
        epochs=epochs,
        balance=balance,
        dst_val=val_dataset,
        lam=merged_args.lam,
        args=merged_args,
        acf_mapping=acf_mapping,
    )

    selected_indices = gradmatch.select()["indices"]
    print(f"Selected {len(selected_indices)} samples using GradMatch.")
    return np.array([train_dataset.data.iloc[i]["uid"] for i in selected_indices])

def load_uids_with_zcore(
    fraction=0.25,
    args=None
) -> np.ndarray:
    p = Path(args.val_embedding_path) # something like 'all_datasets/iWildCam/embeddings/val1_embeddings.npy'
    filename = p.name
    dataset_dir = p.parent.parent.name
    dataset_name = dataset_dir

    train_dataset = get_dataset(dataset_name, split="train")

    embedding_path = args.embedding_path
    embeddings = np.load(embedding_path, allow_pickle=True)

    args_dict = {
        "trial": 0,
        "dataset": dataset_name,
        "num_workers": 10,
        "rand_init": True,
        "redund_exp": 4,
    }
    
    user_args = vars(args)

    custom_args = ['n_sample', 'redund_nn', 'sample_dim']
    for arg in custom_args:
        if arg in user_args:
            user_args[arg] = int(user_args[arg])

    merged_dict = {**user_args, **args_dict}
    args = argparse.Namespace(**merged_dict)

    print("args:", args)

    scores = cs.zcore_score(args, embeddings)

    k = int(fraction * len(train_dataset))
    top = np.argpartition(scores, -k)
    selected_indices = top[-k:]

    selected_indices = selected_indices[selected_indices < len(train_dataset)]
    selected_uids = [train_dataset.data.iloc[i]["uid"] for i in selected_indices]

    return selected_uids


def load_uids_with_glister(
    fraction=0.25,
    random_seed=42,
    epochs=50,
    balance=True,
    eta=0.1,
    args=None
) -> np.ndarray:
    train_dataset, val_dataset, merged_args = _prepare_deepcore_method_inputs(args)

    glister = Glister(
        dst_train=train_dataset,
        fraction=fraction,
        random_seed=random_seed,
        epochs=epochs,
        balance=balance,
        dst_val=val_dataset,
        eta=eta,
        args=merged_args,
    )

    selected_indices = glister.select()["indices"]
    print(f"Selected {len(selected_indices)} samples using Glister.")
    return np.array([train_dataset.data.iloc[i]["uid"] for i in selected_indices])


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

    train_csv_path = None
    val_csv_path = None
    if args.supervised:
        p = Path(args.val_embedding_path) # something like 'all_datasets/iWildCam/embeddings/val1_embeddings.npy'

        dataset_dir = p.parent.parent.name
        dataset_name = dataset_dir
        cfg = DATASETS[dataset_name]

        csv_root = cfg["csv_root_path"]
        filepaths = cfg["FILEPATHS"]

        train_csv_path = str(Path(csv_root) / filepaths["train"])
        
        filename = p.name
        val_split = filename.split('_')[0]
        # val_csv_path = f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{dataset_name}/redone_splits/{val_split}_id.csv"
        val_csv_path = str(Path(csv_root) / filepaths[val_split])

    # for latency benchmarking
    start_time = None
    end_time = None

    if args.name == "no_filter":
        uids = load_uids(
            args.embedding_path,
            train_csv_path,
            val_csv_path,
        )
    # elif args.name == "basic_filter":
    #     uids = load_uids_with_basic_filter(
    #         args.embedding_path,
    #         args.num_workers,
    #     )
    elif args.name == "random_filter":
        uids = load_uids_with_random_filter(
            embedding_path=args.embedding_path,
            subset_percent=args.fraction,
            random_seed=args.random_seed,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
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
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
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
            pool_embedding_path=args.embedding_path,
            fraction=args.fraction,
            random_seed=args.random_seed,
            train_csv_path = train_csv_path,
            val_csv_path = val_csv_path
        )
    elif args.name in ["gradmatch", "gradmatch_acf"]:
        if hasattr(args, "time_path") and args.time_path:
            start_time = time.perf_counter()
        
        uids = load_uids_with_gradmatch(
            fraction=args.fraction, 
            random_seed=args.random_seed,
            balance=True, 
            epochs=int(args.num_epochs),
            args=args,
        )

        if hasattr(args, "time_path") and args.time_path:
            end_time = time.perf_counter()

    elif args.name == "zcore":
        uids = load_uids_with_zcore(
            fraction=args.fraction,
            args=args,
        )
    elif args.name == "glister":
        if hasattr(args, "time_path") and args.time_path:
            start_time = time.perf_counter()

        uids = load_uids_with_glister(
            fraction=args.fraction, 
            random_seed=args.random_seed,
            balance=True,
            epochs=int(args.num_epochs),
            args=args,
        )

        if hasattr(args, "time_path") and args.time_path:
            end_time = time.perf_counter()
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    # save execution time
    if start_time is not None and end_time is not None:
        if hasattr(args, "time_path") and not os.path.exists(args.time_path):
            time_elapsed = end_time - start_time
            print("Saving latency:", time_elapsed, "to path", args.time_path)
            with open(args.time_path, "w") as f:
                f.write(str(time_elapsed) + "\n")


    if os.path.exists(args.save_path):
        return

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
    
    

    

    

