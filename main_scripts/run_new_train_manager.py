import os
import subprocess
import yaml

from utils import *
from manager_wrapper import run_manager

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_NEW_TRAIN = ROOT_DIR / "baselines/run_new_train.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

# --- Configuration ---
# Subset selection methods to evaluate
baselines_list = ["no_filter", "random_filter", "clip_score", "match_dist", "gradmatch_acf", "gradmatch", "glister", "tsds"]

# Parameter sweep loaded from config.yaml
sweep_dict = create_sweep_dict()

# (dataset, val_split, test_split) -- uncomment to add datasets/splits
dataset_list = [
    ('iWildCam', 'val1', 'test1'),
    ('iWildCam', 'val2', 'test2'),
    ('iWildCam', 'val3', 'test3'),
    ('iWildCam', 'val4', 'test4'),
    ('AutoArborist', 'val1', 'test1'),
    ('AutoArborist', 'val2', 'test2'),
    ('AutoArborist', 'val3', 'test3'),
    ('AutoArborist', 'val4', 'test4'),
    ('GeoDE', 'val1', 'test1'),
    ('GeoDE', 'val2', 'test2'),
    ('GeoDE', 'val3', 'test3'),
    ('GeoDE', 'val4', 'test4'),
]

# Training settings to sweep over
finetune_list = ["lora_finetune_vit", "full_finetune_resnet50"]
lr_list = [0.001]
batch_size_list = [128]

with open(str(DATASETS_CONFIG), 'r') as file:
    datasets_config = yaml.safe_load(file)


def create_commands():
    jobs = []
    total_subsets_not_created = 0

    for baseline in baselines_list:
        print("=" * 50)
        print(f"Tuning method params for {baseline}")
        params = sweep_dict[baseline]
        for param_setting in get_sweep_combinations(params, baseline):
            print("Trying param configuration:", param_setting)

            for dataset, val_split, test_split in dataset_list:
                for finetune_type in finetune_list:
                    for lr in lr_list:
                        for batch_size in batch_size_list:
                            save_folder = create_save_folder(dataset, baseline, param_setting)

                            subset_path = save_folder + f"{test_split}_subset.npy"

                            metrics_path = save_folder + f"{test_split}_{finetune_type}_lr={lr}_batchsize={batch_size}_metrics.json"
                            checkpoint_path = save_folder + f"{test_split}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
                            training_task = datasets_config[dataset]["training_task"]

                            if not os.path.exists(subset_path):
                                total_subsets_not_created += 1

                            if not os.path.exists(metrics_path) and os.path.exists(subset_path):
                                command = [str(RUN_NEW_TRAIN), dataset, subset_path, save_folder, "configs/datasets.yaml", str(lr), finetune_type, str(batch_size), checkpoint_path, training_task]
                                jobs.append(command)
                                print("Added command to jobs_to_submit:", " ".join(command))

    print(f"{baselines_list = }")
    print(f"{dataset_list = }")
    print(f"{finetune_list = }")
    print(f"total_jobs = {len(jobs)}")
    print(f"number of subsets not created: {total_subsets_not_created // len(finetune_list)}")

    return jobs


if __name__ == "__main__":
    MAX_QUEUE_SIZE = 10        # max concurrent jobs in SLURM queue
    POLL_INTERVAL = 5 * 60     # seconds between queue checks
    TARGET_JOB_NAME = "run_new_train.sh"  # SLURM job name to track

    jobs_to_submit = create_commands()

    run_manager(MAX_QUEUE_SIZE, POLL_INTERVAL, TARGET_JOB_NAME, jobs_to_submit, run=False)
