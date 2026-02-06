import os
import subprocess
from itertools import product

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_NEW_TRAIN = ROOT_DIR / "baselines/run_new_train.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

BASELINES = ["no_filter", "random_filter", "clip_score", "match_dist", "gradmatch_acf", "gradmatch", "glister", "tsds"]

FINETUNE_TYPES = ["lora_finetune_vit", "full_finetune_resnet50"]
LR_LIST = [0.001]
BATCH_SIZE_LIST = [128]

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

sweep_dict = create_sweep_dict()

with open(str(DATASETS_CONFIG), 'r') as file:
    datasets_config = yaml.safe_load(file)

total_jobs = 0
missing_subsets = 0

for baseline in BASELINES:
    print("=" * 50)
    print(f"Tuning method params for {baseline}")
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params, baseline):
        for dataset, val_split, test_split in DATASET_LIST:
            save_folder = create_save_folder(dataset, baseline, param_setting)
            subset_path = save_folder + f"{test_split}_subset.npy"

            if not os.path.exists(subset_path):
                missing_subsets += 1
                continue

            training_task = datasets_config[dataset]["training_task"]

            for finetune_type, lr, batch_size in product(FINETUNE_TYPES, LR_LIST, BATCH_SIZE_LIST):
                metrics_path    = save_folder + f"{test_split}_{finetune_type}_lr={lr}_batchsize={batch_size}_metrics.json"
                checkpoint_path = save_folder + f"{test_split}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"

                if not os.path.exists(metrics_path):
                    command = [
                        "sbatch", str(RUN_NEW_TRAIN), dataset, subset_path, save_folder,
                        "configs/datasets.yaml", str(lr), finetune_type,
                        str(batch_size), checkpoint_path, training_task,
                    ]
                    total_jobs += 1
                    subprocess.call(command)

print(f"{total_jobs = }")
print(f"subsets not yet created: {missing_subsets}")
