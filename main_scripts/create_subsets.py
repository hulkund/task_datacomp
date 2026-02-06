import os
import subprocess

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"
RUN_CSV_BASELINE = ROOT_DIR / "run_csv_baseline.sh"

# --- Configuration ---
# Subset selection methods to run
BASELINES = ["no_filter", "random_filter", "clip_score", "match_dist", "tsds", "gradmatch", "gradmatch_acf", "glister"]
# Methods that take CSV input instead of embeddings
CSV_BASELINES = {"match_dist", "match_label"}
# DeepCore methods that require a warmstart checkpoint
WARMSTART_BASELINES = {"gradmatch", "gradmatch_acf", "glister"}

# Warmstart training settings for DeepCore methods
NUM_EPOCHS = 50
MODEL_ARCH = "ResNet18"
SUPERVISED = True
USE_PRETRAINED_WARMSTART = True

# (dataset, val_split, test_split)
DATASET_LIST = [
    ('iWildCam', 'val1', 'test1'),
    # ('iWildCam', 'val2', 'test2'),
    # ('iWildCam', 'val3', 'test3'),
    # ('iWildCam', 'val4', 'test4'),
    # ('AutoArborist', 'val1', 'test1'),
    # ('AutoArborist', 'val2', 'test2'),
    # ('AutoArborist', 'val3', 'test3'),
    # ('AutoArborist', 'val4', 'test4'),
    # ('GeoDE', 'val1', 'test1'),
    # ('GeoDE', 'val2', 'test2'),
    # ('GeoDE', 'val3', 'test3'),
    # ('GeoDE', 'val4', 'test4'),
]

# Parameter sweep loaded from config.yaml
sweep_dict = create_sweep_dict()

total_jobs = 0

for baseline in BASELINES:
    print("=" * 50)
    print(f"Tuning method params for {baseline}")
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params, baseline):
        for dataset, val_split, test_split in DATASET_LIST:
            save_folder = create_save_folder(dataset, baseline, param_setting)
            subset_path = save_folder + f"{test_split}_subset.npy"

            if os.path.exists(subset_path):
                continue

            assert "fraction" in param_setting
            fraction = str(param_setting["fraction"])
            random_seed = str(param_setting["random_seed"])

            if baseline in CSV_BASELINES:
                task_num = test_split[4]
                command = [
                    "sbatch", str(RUN_CSV_BASELINE), baseline, dataset, task_num,
                    fraction, subset_path, random_seed,
                ]
            else:
                embedding_path     = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                centroids_path     = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                val_embedding_path = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"

                command = [
                    "sbatch", str(RUN_BASELINE), baseline, embedding_path, subset_path,
                    fraction, val_embedding_path, centroids_path, str(SUPERVISED), random_seed,
                ]

                for k, v in param_setting.items():
                    if k == "fraction":
                        continue
                    command += [f"--{k}", str(v)]

                if baseline in WARMSTART_BASELINES:
                    ckpt_dir = create_warmstart_ckpt_dir(
                        dataset=dataset,
                        val_split=val_split,
                        method=baseline,
                        model=MODEL_ARCH,
                        num_epochs=NUM_EPOCHS,
                        random_seed=random_seed,
                    )

                    # for using the warmstart checkpoint
                    command += [
                        "--model", MODEL_ARCH,
                        "--num_epochs", str(NUM_EPOCHS),
                        "--use_pretrained_warmstart", str(USE_PRETRAINED_WARMSTART),
                        "--warmstart_ckpt_dir", ckpt_dir
                    ]

            total_jobs += 1
            print(subset_path)
            subprocess.call(command)

print(f"{total_jobs = }")
