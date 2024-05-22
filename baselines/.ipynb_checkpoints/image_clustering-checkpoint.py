"""
This is a command line script for clustering image embeddings for the DataComp pool.
The output of the script is a numpy file containing the computed cluster centers.
Please see image_based_clustering.md for additional information, and note that we also provide precomputed numpy files with the cluster centers used in the DataComp baselines.
"""

import argparse
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from typing import Any, List, Tuple

import faiss
import fasttext
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# from baselines.apply_filter import caption_filter
# from baselines.utils import download, random_seed

torch.backends.cudnn.benchmark = True


def train_kmeans(
    embeddings: np.ndarray, num_clusters: int, num_gpus: int, seed: int = 0
) -> torch.Tensor:
    """train kmeans on embeddings

    Args:
        embeddings (np.ndarray): embeddings to cluster
        num_clusters (int): number of clusters
        num_gpus (int): number of gpus to use
        seed (int, optional): random seed. Defaults to 0.
    """
    d = embeddings.shape[1]
    cluster = faiss.Clustering(d, num_clusters)
    cluster.verbose = True
    cluster.niter = 20
    cluster.seed = seed

    # otherwise the kmeans implementation sub-samples the training set
    cluster.max_points_per_centroid = 5000

    res = [faiss.StandardGpuResources() for i in range(num_gpus)]

    flat_config = []
    for i in range(num_gpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if num_gpus == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [
            faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(num_gpus)
        ]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    cluster.train(embeddings, index)
    centroids = faiss.vector_float_to_array(cluster.centroids)

    return centroids.reshape(num_clusters, d)


# def load_embedding_helper(
#     fs_root: str,
#     key: str = "image_embedding",
#     caption_filtering: bool = False,
#     sample_ratio: float = -1.0,
# ) -> np.ndarray:
#     """worker function to load embeddings

#     Args:
#         fs_root (Tuple[Any, str]): (filesystem, path_root)
#         key (str, optional): key to load from npz. Defaults to "l14_img".
#         caption_filtering (bool, optional): whether to enable caption filter. Defaults to False.
#         sample_ratio (float, optional): ratio of samples to use. Defaults to -1.0.
#     """

#     # fs, path_root = fs_root
#     embed = np.load(f"{fs_root}",allow_pickle=True)#[0][key]
#     embed = np.vstack([e[key] for e in embed])
#     print(embed.shape)
#     if caption_filtering:
#         lang_detect_model = fasttext.load_model(
#             download("fasttext", "~/.cache/fasttext")
#         )
#         df = pd.read_parquet(
#             f"{path_root}.parquet", columns=["uid", "text"], filesystem=fs
#         )
#         mask = caption_filter(df, lang_detect_model)
#         embed = embed[mask]
#     if sample_ratio > 0:
#         n = len(embed)
#         idx = np.random.choice(range(n), size=int(n * sample_ratio))
#         embed = embed[idx]
#     return embed


def load_embedding(
    path: str,
    n_workers: int = 10,
    key: str = "image_embedding",
    caption_filtering: bool = False,
    sample_ratio: float = -1.0,
) -> np.ndarray:
    """worker function to load embeddings

    Args:
        paths (List[Tuple[Any, str]]): list of (filesystem, path_root)
        n_workers (int, optional): number of workers. Defaults to 10.
        key (str, optional): key to load from npz. Defaults to "l14_img".
        caption_filtering (bool, optional): whether to enable caption filter. Defaults to False.
        sample_ratio (float, optional): ratio of samples to use. Defaults to -1.0.
    """
    mp.set_start_method("spawn", force=True)
    print("start loading embedding")
    embed = np.load(f"{path}",allow_pickle=True)#[0][key]
    embed = np.vstack([e[key] for e in embed])
    print("embedding shape", embed.shape)
    if caption_filtering:
        lang_detect_model = fasttext.load_model(
            download("fasttext", "~/.cache/fasttext")
        )
        df = pd.read_parquet(
            f"{path_root}.parquet", columns=["uid", "text"], filesystem=fs
        )
        mask = caption_filter(df, lang_detect_model)
        embed = embed[mask]
    if sample_ratio > 0:
        n = len(embed)
        idx = np.random.choice(range(n), size=int(n * sample_ratio))
        embed = embed[idx]
    return embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--metadata_dir",
    #     type=str,
    #     help="directory (local or cloud) containing parquet, npz metadata",
    # )
    # parser.add_argument("--save_path", type=str, help="local path to output centroids")
    parser.add_argument(
        "--dataset_name", default="COOS", type=str, help="name of dataset to cluster"
    )
    parser.add_argument(
        "--num_clusters", default=1000, type=int, help="number of clusters"
    )
    parser.add_argument(
        "--sample_ratio",
        default=-1.0,
        type=float,
        help="ratio of samples to use (we need to sample because of memory constraint)",
    )
    parser.add_argument(
        "--num_gpus", default=1, type=int, help="number of gpus used for clustering"
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers, generally set to number of cpu cores",
    )
    parser.add_argument(
        "--disable_caption_filtering",
        default=True,
        action="store_true",
        help="whether to disable text-based basic filtering",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    # random_seed(args.seed)

    num_clusters = args.num_clusters
    num_gpus = args.num_gpus
    sample_ratio = args.sample_ratio
    caption_filtering = not args.disable_caption_filtering
    # fs, url = fsspec.core.url_to_fs(args.metadata_dir)
    # paths = [(fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x]
    root_path = f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{args.dataset_name}/embeddings/"
    splits = ["train","val1","val2","val3","val4"]
    # paths = [root_path+"COOS_{}_embeddings.npy".format(split) for split in splits]
    for split in splits:
        if "val" in split:
            num_clusters=50
        path = root_path+"{}_embeddings.npy".format(split) 
        print(f"split: {split} | caption filtering: {caption_filtering} | sample_ratio={sample_ratio}")
        embeddings = load_embedding(
            path,
            key="image_embedding",
            n_workers=args.num_workers,
            caption_filtering=caption_filtering,
            sample_ratio=sample_ratio,
        )
        print(f"done: {len(embeddings)}")
    
        print(f"start clustering: num_clusters = {num_clusters}, num_gpus = {num_gpus}")
        embeddings = embeddings.astype(np.float32)
        centroids = train_kmeans(
            embeddings, num_clusters, num_gpus=num_gpus, seed=args.seed
        )
        save_path = f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{args.dataset_name}/centroids/"+"{}_centroids.pt".format(split)
        torch.save(centroids, save_path, pickle_protocol=4)
