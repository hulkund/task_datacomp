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
import sklearn

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
    # if fraction is not None:
    #     threshold, _ = get_threshold(embedding_path=pool_embedding_path, 
    #                               key=sim_key, 
    #                               fraction=fraction)
    #     uids=np.array([uid for uid in pool_embed_df[pool_embed_df[sim_key] >= threshold]["uid"]])
    #     pool_embed_df = pool_embed_df[pool_embed_df["uid"].isin(uids)] 
    
    # caption filter - nEED TO ADD
    # mask = caption_filter(df, lang_detect_model)
    # uids = pool_embed_df.uid[mask]
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

# def image_filter_helper(
#     pool_centroids: torch.Tensor,
#     target_centroid_ids: torch.Tensor,
#     batch_size: int,
#     threshold: Union[float, None] = None,
# )



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
    

# THEN AT THE END, THE MAIN FUNCTION
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

    if args.name == "no_filter":
        uids = load_uids(
            args.embedding_path
        )
    elif args.name == "basic_filter":
        uids = load_uids_with_basic_filter(
            args.embedding_path,
            args.num_workers,
        )
    elif args.name == "random_filter":
        uids = load_uids_with_random_filter(
            embedding_path=args.embedding_path,
            subset_percent=args.fraction
        )
    # get rid of this shit
    # elif args.name == "text_based":
    #     nltk.download("wordnet")
    #     uids = load_uids_with_text_entity(
    #         args.metadata_dir,
    #         args.num_workers,
    #     )
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
    else:
        raise ValueError(f"Unknown args.name argument: {args.name}")

    print(f"sorting {len(uids)} uids")
    uids.sort()

    print(f"saving {args.save_path} with {len(uids)} entries")
    directory = os.path.dirname(args.save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(args.save_path, uids)
    
    

    

    

