import numpy as np
import sklearn.metrics
from utils import load_embedding


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