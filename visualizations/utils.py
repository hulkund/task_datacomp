import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union

train_csv_path = "/data/vision/beery/scratch/neha/task-datacomp/all_datasets/iWildCam/new_splits/train.csv"

UID_COL = "uid"
LABEL_COL = "label"

def plot_train_and_subsets(
    subset_paths: Union[str, List[str]],
    train_csv_path: str = train_csv_path,
    uid_col: str = UID_COL,
    label_col: str = LABEL_COL,
    top_n: int = 50,
    colors: List[str] = None,
):
    """
    Plots label frequency distributions for the training set and one or more subsets.
    The title for each subset plot is automatically set to the name of the directory
    containing the .npy file (for clarity when comparing multiple runs).

    Parameters
    ----------
    subset_paths : str | list[str]
        Path or list of paths to .npy files containing subset UIDs.
    train_csv_path : str, optional
        Path to the training CSV containing metadata with UID and label columns.
    uid_col : str, optional
        Column name in the CSV representing the UID. Defaults to "uid".
    label_col : str, optional
        Column name in the CSV representing the label/class. Defaults to "category_id".
    top_n : int, optional
        Number of top most frequent labels to plot. Defaults to 50.
    colors : list[str], optional
        List of bar colors for the plots. Automatically assigned if not provided.
    """
    # Normalize input type
    if isinstance(subset_paths, str):
        subset_paths = [subset_paths]

    n_subsets = len(subset_paths)

    # Load training CSV
    train_df = pd.read_csv(train_csv_path)

    # Compute training label frequencies
    train_label_counts = train_df[label_col].value_counts()
    sorted_labels = train_label_counts.index.tolist()
    top_labels = sorted_labels[:top_n]
    train_freqs = [train_label_counts.get(lbl, 0) for lbl in top_labels]

    # Load subsets
    subset_counts = []
    for path in subset_paths:
        subset_uids = np.load(path, allow_pickle=True)
        subset_df = train_df[train_df[uid_col].isin(subset_uids)]
        subset_label_counts = subset_df[label_col].value_counts()
        freqs = [subset_label_counts.get(lbl, 0) for lbl in top_labels]

        # Extract descriptive folder name
        subset_name = os.path.basename(os.path.dirname(path))
        subset_counts.append((subset_name, freqs))

    # ✅ Fix: ensure colors are lists, not tuples
    if colors is None:
        colors = ["gray"] + list(plt.cm.tab10.colors[:n_subsets])
    else:
        colors = ["gray"] + list(colors)

    # --- Plot ---
    fig, axes = plt.subplots(1 + n_subsets, 1, figsize=(14, 4 * (1 + n_subsets)), sharex=True)

    # Plot training distribution
    axes[0].bar(range(len(top_labels)), train_freqs, color=colors[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Training Pool class frequency", fontsize=14)
    axes[0].set_ylabel("# of Images", fontsize=12)
    axes[0].grid(False)

    # Plot subsets
    for i, (subset_name, freqs) in enumerate(subset_counts, start=1):
        axes[i].bar(range(len(top_labels)), freqs, color=colors[i % len(colors)])
        axes[i].set_yscale("log")
        axes[i].set_title(f"Subset {i}: {subset_name}", fontsize=13)
        axes[i].set_ylabel("# of Images", fontsize=12)
        axes[i].grid(False)

    # Shared x-axis
    axes[-1].set_xlabel(f"Label", fontsize=12)
    axes[-1].set_xticks(range(len(top_labels)))
    axes[-1].set_xticklabels(top_labels, rotation=90)

    plt.tight_layout()
    plt.show()


def plot_test_and_subsets(
    test_csv_path: str,
    train_csv_path: str,
    subset_paths: Union[str, List[str]],
    uid_col: str = UID_COL,
    label_col: str = LABEL_COL,
    top_n: int = 50,
    colors: List[str] = None,
):
    """
    Plots label frequency distributions for a test CSV and one or more subset .npy files.
    Uses train.csv only to map UIDs → labels for the subsets.

    Parameters
    ----------
    test_csv_path : str
        Path to the CSV representing the test dataset (reference histogram).
    train_csv_path : str
        Path to the training CSV with UID → label mapping.
    subset_paths : str | list[str]
        Path or list of paths to .npy files containing subset UIDs.
    uid_col : str, optional
        Column name for the UID. Defaults to "UID".
    label_col : str, optional
        Column name for the label. Defaults to "label".
    top_n : int, optional
        Number of top most frequent labels to display. Defaults to 50.
    colors : list[str], optional
        Custom list of colors for the bars. Automatically assigned if None.
    """

    # Normalize subset paths
    if isinstance(subset_paths, str):
        subset_paths = [subset_paths]

    n_subsets = len(subset_paths)

    # --- Load data ---
    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path)

    if label_col not in test_df.columns or label_col not in train_df.columns:
        raise ValueError(f"Both CSVs must contain a '{label_col}' column.")

    # --- Compute label frequencies for test set ---
    test_label_counts = test_df[label_col].value_counts()
    top_labels = test_label_counts.index[:top_n]
    test_freqs = [test_label_counts.get(lbl, 0) for lbl in top_labels]

    # --- Compute label frequencies for each subset using train.csv mapping ---
    subset_counts = []
    for path in subset_paths:
        subset_uids = np.load(path, allow_pickle=True)
        subset_df = train_df[train_df[uid_col].isin(subset_uids)]
        subset_label_counts = subset_df[label_col].value_counts()
        freqs = [subset_label_counts.get(lbl, 0) for lbl in top_labels]
        subset_name = os.path.basename(os.path.dirname(path))
        subset_counts.append((subset_name, freqs))

    # --- Colors ---
    if colors is None:
        colors = ["gray"] + list(plt.cm.tab10.colors[:n_subsets])
    else:
        colors = ["gray"] + list(colors)

    # --- Plot ---
    fig, axes = plt.subplots(1 + n_subsets, 1, figsize=(14, 4 * (1 + n_subsets)), sharex=True)

    # Plot test set histogram
    axes[0].bar(range(len(top_labels)), test_freqs, color=colors[0])
    axes[0].set_yscale("log")
    axes[0].set_title(f"Deployment 1 class frequency: {os.path.basename(test_csv_path)}", fontsize=14)
    axes[0].set_ylabel("# of Images", fontsize=12)
    axes[0].grid(False)

    # Plot subsets
    for i, (subset_name, freqs) in enumerate(subset_counts, start=1):
        axes[i].bar(range(len(top_labels)), freqs, color=colors[i % len(colors)])
        if any(f > 0 for f in freqs):
            axes[i].set_yscale("log")
        else:
            axes[i].set_yscale("linear")
        axes[i].set_title(f"Subset {i}: {subset_name}", fontsize=13)
        axes[i].set_ylabel("# of Images", fontsize=12)
        axes[i].grid(False)

    # Shared x-axis labels
    axes[-1].set_xlabel(f"Label", fontsize=12)
    axes[-1].set_xticks(range(len(top_labels)))
    axes[-1].set_xticklabels(top_labels, rotation=90)

    plt.tight_layout()
    plt.show()
