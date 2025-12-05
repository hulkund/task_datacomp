import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union

from visualizations.file_utils import create_confusion_matrix_png_save_path, create_per_class_accuracy_json_save_path
from baselines.utils import get_dataset

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


def get_class_accuracies(pt_path: str, save_json: bool = False) -> dict:
    data = torch.load(pt_path, map_location="cpu")
    labels = np.array(data["labels"])
    preds  = np.array(data["preds"])
    mapping = data["mapping"]

    class_acc = {}
    for true_c in mapping:
        mapped_c = mapping[true_c]
        idx = (labels == mapped_c)
        if idx.sum() > 0:
            class_acc[true_c] = (preds[idx] == labels[idx]).mean()

    if save_json:
            json_save_path = create_per_class_accuracy_json_save_path(pt_path)
            with open(json_save_path, "w") as jf:
                json.dump(class_acc, jf, indent=2)
            print(f"Class accuracy dictionary saved to: {json_save_path}")
    
    return class_acc


def plot_confusion_matrix_from_pt(
    pt_path: str,
    normalize: bool = True,
    annot=False,
    figsize: tuple = (8, 6),
    cmap: str = "Blues",
    save_png: bool = True,
):
    """
    Load a saved logits .pt file and plot its confusion matrix.

    Args:
        pt_path (str): Path to the .pt file containing logits, labels, preds, and mapping.
        normalize (bool): Whether to normalize confusion matrix rows (true label fractions).
        annot (bool): Whether to add the annotation to each cell.
        figsize (tuple): Size of the matplotlib figure.
        cmap (str): Colormap used for the heatmap.
        save_png (bool): If true, saves the confusion matrix figure to confusion_matrix.png.

    The .pt file must contain:
        - 'logits': torch.Tensor
        - 'labels': torch.Tensor or list[int]
        - 'preds': torch.Tensor or list[int]
        - 'mapping': dict mapping class_name -> class_index
    """

    data = torch.load(pt_path, map_location="cpu")
    labels = np.array(data["labels"])
    preds  = np.array(data["preds"])
    mapping = data["mapping"]

    num_classes = len(mapping)

    # ---- Confusion matrix ----
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        conf_mat[t, p] += 1

    present_classes = np.unique(labels)
    conf_mat_subset = conf_mat[np.ix_(present_classes, present_classes)]

    scientific_name_mapping = get_dataset('iWildCam', 'train').mapping
    csv = "/data/vision/beery/scratch/neha/task-datacomp/OT_project/relabeling_algorithms/remapping.csv"
    common_name_mapping = pd.read_csv(csv, header=None)
    # common_name_mapping = {k.lower() : v.lower() for k, v in common_name_mapping.items()}
    common_name_mapping = dict(zip(common_name_mapping[0].str.lower(),common_name_mapping[1].str.lower()))

    class_names = [common_name_mapping[scientific_name_mapping[list(mapping.keys())[i]]] for i in present_classes]

    # Normalize rows (true label distribution)
    if normalize:
        row_sums = conf_mat_subset.sum(axis=1, keepdims=True)
        conf_mat_plot = np.divide(
            conf_mat_subset, row_sums,
            out=np.zeros_like(conf_mat_subset, dtype=float),
            where=row_sums != 0
        )
        fmt = ".2f"
        cbar_label = "Fraction of True Class"
    else:
        conf_mat_plot = conf_mat_subset
        fmt = "d"
        cbar_label = "Count"

    # ---- Plot ----
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_mat_plot,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': cbar_label}
    )
    plt.title(f"{'Normalized' if normalize else 'Raw'} Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    if save_png:
        png_save_path = create_confusion_matrix_png_save_path(pt_path)
        plt.savefig(png_save_path, dpi=300)
        print(f"\nConfusion matrix saved to {png_save_path}")
    
    plt.show()


from collections import defaultdict
from matplotlib_venn import venn2, venn3


class UIDSelectorComparator:
    def __init__(self, uid_files: dict, label_map_file=None):
        """
        uid_files: dict(name -> path_to_npy)
            e.g. {
                "gradmatch": "path/.../gradmatch.npy",
                "tsds": "path/.../tsds.npy"
            }

        label_map_file: Optional path to a .npy or .pkl mapping
            UID -> label (integer).
        """
        self.uid_files = uid_files
        self.methods = list(uid_files.keys())
        self.uid_sets = {k: set(np.load(v, allow_pickle=True)) for k, v in uid_files.items()}

        if label_map_file is not None:
            self.uid_to_label = np.load(label_map_file, allow_pickle=True).item()
        else:
            self.uid_to_label = None

    # ---------------------------------------------------------
    # Core metrics
    # ---------------------------------------------------------
    def iou(self, method1, method2):
        A, B = self.uid_sets[method1], self.uid_sets[method2]
        return len(A & B) / len(A | B)

    def overlap_counts(self, method1, method2):
        A, B = self.uid_sets[method1], self.uid_sets[method2]
        return {
            "intersection": len(A & B),
            "only_method1": len(A - B),
            "only_method2": len(B - A),
            "union": len(A | B)
        }

    # ---------------------------------------------------------
    # Per-label metrics
    # ---------------------------------------------------------
    def per_label_iou(self, method1, method2):
        if self.uid_to_label is None:
            raise ValueError("Need a UID→label mapping to compute per-label IOU")

        # bucket uids by label
        label_buckets = defaultdict(lambda: defaultdict(set))

        for m in [method1, method2]:
            for uid in self.uid_sets[m]:
                lbl = self.uid_to_label[uid]
                label_buckets[lbl][m].add(uid)

        results = {}
        for lbl, d in label_buckets.items():
            A = d.get(method1, set())
            B = d.get(method2, set())
            if len(A) + len(B) == 0:
                results[lbl] = 0.0
            else:
                results[lbl] = len(A & B) / len(A | B)

        return results

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    def venn(self, method1, method2):
        A = self.uid_sets[method1]
        B = self.uid_sets[method2]

        plt.figure(figsize=(6, 6))
        venn2([A, B], set_labels=(method1, method2))
        plt.title(f"UID Overlap: {method1} vs {method2}")
        plt.show()

    def venn_three(self, m1, m2, m3):
        A = self.uid_sets[m1]
        B = self.uid_sets[m2]
        C = self.uid_sets[m3]

        plt.figure(figsize=(7, 7))
        venn3([A, B, C], set_labels=(m1, m2, m3))
        plt.title(f"UID Overlap Across Methods")
        plt.show()

    def barplot_per_label_iou(self, method1, method2):
        per_label = self.per_label_iou(method1, method2)
        labels = sorted(per_label.keys())
        values = [per_label[l] for l in labels]

        plt.figure(figsize=(12, 5))
        plt.bar(labels, values)
        plt.xlabel("Label")
        plt.ylabel("IOU")
        plt.title(f"Per-Label IOU: {method1} vs {method2}")
        plt.show()

    # ---------------------------------------------------------
    # Qualitative visualization
    # ---------------------------------------------------------
    def visualize_examples(self, method1, method2, image_loader, k=5):
        """
        image_loader: function(uid) -> image array (H,W,3)
        k: number of samples to show per category
        """
        A = self.uid_sets[method1]
        B = self.uid_sets[method2]

        only1 = list(A - B)[:k]
        only2 = list(B - A)[:k]
        both = list(A & B)[:k]

        groups = [("Only " + method1, only1),
                  ("Both", both),
                  ("Only " + method2, only2)]

        for title, uids in groups:
            if not uids:
                continue

            plt.figure(figsize=(15, 3))
            for i, uid in enumerate(uids):
                img = image_loader(uid)
                plt.subplot(1, k, i + 1)
                plt.imshow(img)
                plt.title(uid)
                plt.axis("off")
            plt.suptitle(title)
            plt.show()

    def summary_stats(self, method1, method2):
        A = self.uid_sets[method1]
        B = self.uid_sets[method2]

        inter = len(A & B)
        union = len(A | B)
        only1 = len(A - B)
        only2 = len(B - A)

        print(f"=== Summary: {method1} vs {method2} ===")
        print(f"{method1} count: {len(A)}")
        print(f"{method2} count: {len(B)}")
        print(f"Intersection: {inter}")
        print(f"Only {method1}: {only1}")
        print(f"Only {method2}: {only2}")
        print(f"Union: {union}")
        print(f"IOU: {inter / union:.4f}")
        print(f"Overlap (intersection / min(n1,n2)): {inter / min(len(A), len(B)):.4f}")
        print("====================================")