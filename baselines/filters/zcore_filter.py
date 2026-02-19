import argparse
import numpy as np

from zcore.core.coreset import zcore as zcore_module
from baselines.filters.utils import load_embedding

# TODO: n_sample is 1_000 for testing, but 1_000_000 in paper
def load_uids_with_zcore(
    pool_embedding_path: str,
    fraction: float,
    n_sample: int = 1_000,
    sample_dim: int = 2,
    rand_init: bool = True,
    redund_nn: int = 1000,
    redund_exp: int = 4,
    num_workers: int = 2,
    key: str = "image_embedding",
) -> np.ndarray:

    # load image embeddings (TODO: perhaps later concatenate more embeddings like in ZCore paper)
    key = "image_embedding"
    pool_embed = load_embedding(pool_embedding_path, [key, "uid"])
    # reshape embeddings to (N, D) matrix, as expected by ZCore implementation
    embeddings = np.vstack(pool_embed[key])
    args = {
        'n_sample': n_sample,
        'sample_dim': sample_dim,
        'rand_init': rand_init,
        'redund_nn': redund_nn,
        'redund_exp': redund_exp,
        'num_workers': min(num_workers, 8),
        'dataset': 'task_datacomp', # just a label for logging during ZCore's iterations
        'trial': 0
    }
    args = argparse.Namespace(**args)

    # Select top n_keep datapoints
    scores = zcore_module.zcore_score(args, embeddings) # (N,)
    n_keep = int(len(scores) * fraction)
    top_indices = np.argsort(scores)[-n_keep:]
    return np.array(pool_embed["uid"].iloc[top_indices])
