import os
import subprocess
from itertools import product

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"

# --- Configuration ---
SUPERVISED = True

# Datasets, validation splits, and random seeds to train over
DATASETS = ["iWildCam", "AutoArborist", "GeoDE"]
VAL_SPLITS = ["val1", "val2", "val3", "val4"]
SEEDS = [0, 1, 42]

# Warmstart training settings for DeepCore methods
NUM_EPOCHS = 50
MODEL_ARCH = "ResNet18"
BASELINE = "gradmatch"

# Gradmatch hyperparameters
SELECTION_BATCH = "16"
SELECTION_LR = "0.01"
LAM = "0.5"

total_jobs = 0

for dataset, val_split, random_seed in product(DATASETS, VAL_SPLITS, SEEDS):
    embedding_path     = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    centroids_path     = f"all_datasets/{dataset}/centroids/train_centroids.pt"
    val_embedding_path = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"

    ckpt_dir = create_warmstart_ckpt_dir(
        dataset=dataset,
        val_split=val_split,
        method=BASELINE,
        model=MODEL_ARCH,
        num_epochs=NUM_EPOCHS,
        random_seed=random_seed,
    )

    ckpt_path = ckpt_dir + "warmstart_weights.pth"
    if not os.path.exists(ckpt_path):
        command = [
            "sbatch", str(RUN_BASELINE), BASELINE, embedding_path, ckpt_path,
            "1", val_embedding_path, centroids_path, str(SUPERVISED), str(random_seed),
            "--selection_batch", SELECTION_BATCH,
            "--selection_lr", SELECTION_LR,
            "--lam", LAM,
            "--model", MODEL_ARCH,
            "--num_epochs", str(NUM_EPOCHS),
            "--use_pretrained_warmstart", str(False),
            "--warmstart_ckpt_dir", ckpt_dir,
        ]
        total_jobs += 1
        subprocess.call(command)

print(f"{total_jobs = }")
