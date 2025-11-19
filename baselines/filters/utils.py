import pandas as pd
import numpy as np

def load_embedding(embedding_path:str, columns):
    """load in metadata from a numpy file
    Args:
        embedding_path (str): path to numpy file -- embedding must be stored as a numpy array of dicts
        columns (List[str]): list of columns to extract from the numpy file
    Returns:
        pd.DataFrame: dataframe containing the requested columns
    """
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

def load_uids(embedding_path: str) -> np.ndarray:
    """helper to read a embedding and load uids

    Args:
        fs_url (Tuple[Any, str]): pair of fsspec file system and parquet url

    Returns:
        np.ndarray: array of uids
    """
    df = load_embedding(embedding_path, columns=["uid"])
    return np.array(df['uid'])