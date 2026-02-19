import numpy as np
from ..utils import load_embedding, get_threshold

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