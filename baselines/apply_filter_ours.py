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

from pathlib import Path

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
        threshold, embed_df = get_threshold(embedding_path, key, fraction)
    else: 
        embed_df = load_embedding(embedding_path)
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

    


def load_embedding(embedding_path:str, columns):
    embed = np.load(f"{embedding_path}",allow_pickle=True)
    embed_df=pd.DataFrame()
    for col in columns:
        embed_df[col] = [e[col] for e in embed]
    return embed_df

def get_threshold(
    embedding_path: str, key: str, fraction: float
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
    embed_df = load_embedding(embedding_path, columns=[key,"uid"])
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


def load_uids(embedding_path: str) -> np.ndarray:
    """helper to read a embedding and load uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    df = load_embedding(embedding_path, columns=["uid"])
    return np.array(df['uid'])

def load_uids_with_random_filter(embedding_path: str, subset_percent: float) -> np.ndarray:
    embed_df=load_embedding(embedding_path, columns=["uid"])
    uids_selected=embed_df.sample(frac=subset_percent,random_state=42)
    return np.array(uids_selected).squeeze(axis=1)

def load_uids_with_tsds(
    val_embedding_path: str, 
    pool_embedding_path: str,
) -> np.ndarray:

    key = "image_embedding"
    query_df=load_embedding(val_embedding_path, [key,"uid"])
    candidate_df=load_embedding(pool_embedding_path, [key,"uid"])
    xq=np.stack(query_df["image_embedding"])
    xb=np.stack(candidate_df["image_embedding"])

    with open("baselines/tsds_data/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    SAMPLE_SIZE = 1000#config["sample_size"]
    MAX_K = config["max_K"]
    KDE_K = config["kde_K"]
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
        # logging.info(f'A point has {(kernel > 0).sum(axis=-1).mean() - 1} near-duplicates on average.')
        kernel = kernel * (kernel > 0)
        kde = kernel.sum(axis=-1)
        kde_map = {top_indices_set[i]:kde[i] for i in range(len(top_indices_set))}
        kde_mapfunc = np.vectorize(lambda t: kde_map[t])
        top_kdes = kde_mapfunc(top_indices)
            
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
    sample_times = np.random.multinomial(SAMPLE_SIZE, global_probs)
    sample_indices = []
    for i in range(sample_times.shape[0]):
        sample_indices.extend([i] * sample_times[i])

    uid_array = candidate_df["uid"].to_numpy()
    selected_uids = np.array([uid_array[i] for i in sample_indices])
    return selected_uids

def load_uids_with_gradmatch(
    fraction=0.25, # sweep over [0.25, 0.5, 0.75, 0.9]
    random_seed=42, 
    epochs=50, # paper uses 200, but we've found that finetuned ResNet overfits on iWildCam after 50 epochs
    balance=True, 
    lam=0.5, # from the paper
    args=None
) -> np.ndarray:
    # TODO: put a bunch of the args_dict into config (or hard-code them here)
    args_dict = {
        'print_freq': 100, 
        'device': 'cuda', 
        'workers': 4, 
        'model':'ResNet18', # from the paper
        'selection_optimizer':'SGD', # from the paper
        'selection_momentum':0.9, # from the paper
        'selection_weight_decay':1e-4, # from the paper
        'selection_nesterov':True,
        'selection_test_interval':10,
        'selection_test_fraction':1.0,
        'specific_model':None,
        'torchvision_pretrain':True,
        'if_dst_pretrain':False,
        'dst_pretrain_dict':{},
        'n_pretrain_size':1000,
        'n_pretrain':1000,
        'gpu':[0]
    }

    # Load datasets using args.dataset_name and args.val_embedding_path
    p = Path(args.val_embedding_path) # something like 'all_datasets/iWildCam/embeddings/val1_embeddings.npy'
    filename = p.name
    dataset_dir = p.parent.parent.name
    dataset_name = dataset_dir
    val_split = filename.split('_')[0]
    print(f"dataset={dataset_name}, val_split={val_split}")
    train_dataset = get_dataset(dataset_name, split='train')
    val_dataset = get_dataset(dataset_name, split=val_split)
    
    # look at what type the dataset label target is
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Example data point: {type(train_dataset[0][0])}")
    print(f"Example label type: {type(train_dataset[0][1])}")

    # TODO: determine these values from train_dataset
    args_dict['num_classes'] = train_dataset.data['label'].nunique()
    args_dict['channel'] = 3
    args_dict['im_size'] = [224,224]

    # Prepare model and args (adapt to your setup)
    # Example: args should have .num_classes, .device, .selection_batch, .print_freq, .workers, etc.

    user_args = vars(args)

    # TODO: can probably cross-reference with config file
    if 'selection_lr' in user_args:
        user_args['selection_lr'] = float(user_args['selection_lr'])
    if 'selection_batch' in user_args:
        user_args['selection_batch'] = int(user_args['selection_batch'])

    merged_dict = {**user_args, **args_dict}
    args = argparse.Namespace(**merged_dict)

    print("Initializing GradMath with args:", vars(args))

    # Initialize GradMatch
    gradmatch = GradMatch(
        dst_train=train_dataset,
        fraction=fraction,
        random_seed=random_seed,
        epochs=epochs,
        balance=balance,
        dst_val=val_dataset,
        lam=lam,
        args=args
    )
    print(gradmatch)
    # Run selection
    selection_result = gradmatch.select()
    selected_indices = selection_result["indices"]
    selected_weights = selection_result["weights"]

    # Filter your dataset using selected_indices
    filtered_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
    print(f"Selected {len(selected_indices)} samples using GradMatch.")

    # Optionally, save indices or filtered dataset
    # np.save("gradmatch_selected_indices.npy", selected_indices)

    selected_uids = np.array([train_dataset.data.iloc[i]["uid"] for i in selected_indices])
    # return filtered_dataset, selected_indices, selected_weights, selected_uids
    return selected_uids

    
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
    
    

    

    

