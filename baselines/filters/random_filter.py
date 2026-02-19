from baselines.filters.utils import load_embedding
import numpy as np
import pandas as pd


def load_uids_with_random_filter(embedding_path: str, subset_percent: float) -> np.ndarray:
    embed_df=load_embedding(embedding_path, columns=["uid"])
    uids_selected=embed_df.sample(frac=subset_percent,random_state=42)
    return np.array(uids_selected).squeeze(axis=1)