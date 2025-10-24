import argparse
import numpy as np
import os
from baselines.utils import get_dataset

import zcore.core.coreset as cs

def get_my_embeddings():
    embed_file = f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/iWildCam/embeddings/train_embeddings.npy"
    model_embed = np.load(embed_file, allow_pickle=True)

    # only look at image embeddings for now
    embeddings = np.vstack([d["image_embedding"] for d in model_embed])
    return embeddings

def main(args, fraction=0.25):
    subset_path = f"/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/zcore_fraction_{fraction}/test1_subset.npy"
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)

    train_dataset = get_dataset('iWildCam', split="train")
    embeddings = get_my_embeddings()
    scores = cs.zcore_score(args, embeddings)

    print("len(train_dataset):", len(train_dataset))
    print("len(embeddings):", len(embeddings))
    print("len(scores):", len(scores))

    k = int(fraction * len(train_dataset))
    top = np.argpartition(scores, -k)
    selected_indices = top[-k:]

    selected_indices = selected_indices[selected_indices < len(train_dataset)]
    selected_uids = [train_dataset.data.iloc[i]["uid"] for i in selected_indices]

    np.save(subset_path, selected_uids)
    print(f"\nZCore score saved at {subset_path}")

args_dict = {
    "trial": 0,
    "dataset": "iWildCam",
    "n_sample": 500,
    "num_workers": 10,
    "rand_init": True,
    "sample_dim": 2,
    "redund_nn": 1000,
    "redund_exp": 4,
}
args = argparse.Namespace(**args_dict)
main(args, fraction=0.25)