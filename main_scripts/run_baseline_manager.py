import os
import subprocess

from utils import *
from manager_wrapper import run_manager

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"
RUN_CSV_BASELINE = ROOT_DIR / "run_csv_baseline.sh"

# --- Configuration ---
# Subset selection methods to run
baselines_list = ["gradmatch", "gradmatch_acf"]

# Parameter sweep loaded from config.yaml
sweep_dict = create_sweep_dict()

# (dataset, val_split, test_split) -- uncomment to add datasets/splits
dataset_list = [
    ('AutoArborist', 'val3', 'test3'),
]

supervised = "True"
use_pretrained_warmstart = "True"


def create_commands():
    jobs = []

    for baseline in baselines_list:
        print("=" * 50)
        print(f"Tuning method params for {baseline}")
        params = sweep_dict[baseline]
        for param_setting in get_sweep_combinations(params, baseline):
            print("Trying param configuration:", param_setting)

            for dataset, val_split, test_split in dataset_list:
                embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                val_embedding_path = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
                save_folder = create_save_folder(dataset, baseline, param_setting)

                # Create subset
                subset_path = save_folder + f"{test_split}_subset.npy"
                if not os.path.exists(subset_path):
                    assert "fraction" in param_setting
                    fraction = str(param_setting["fraction"])
                    random_seed = str(param_setting["random_seed"])

                    if baseline in ["match_dist", "match_label"]:
                        task_num = test_split[4]
                        command = [str(RUN_CSV_BASELINE), baseline, dataset, task_num, fraction, subset_path, random_seed]
                    else:
                        command = [str(RUN_BASELINE), baseline, embedding_path, subset_path, fraction, val_embedding_path, centroids_path, supervised, random_seed]
                        for k, v in param_setting.items():
                            if k == "fraction":
                                continue
                            command.append(f"--{k}")
                            command.append(str(v))

                            command.append("--model")
                            command.append("ResNet18")

                            command.append("--num_epochs")
                            command.append(str(50))

                        command.append("--use_pretrained_warmstart")
                        command.append(use_pretrained_warmstart)

                        if baseline not in ["gradmatch", "gradmatch_acf", "glister"]:
                            ckpt_dir = ""
                        else:
                            ckpt_dir = create_warmstart_ckpt_dir(
                                dataset=dataset,
                                val_split=val_split,
                                method=baseline,
                                model="ResNet18",
                                num_epochs=50,
                                random_seed=random_seed
                            )

                        command.append("--warmstart_ckpt_dir")
                        command.append(ckpt_dir)

                    print("Running command to create subset:", " ".join(command))
                    jobs.append(command)

    print(f"{baselines_list = }")
    print(f"{dataset_list = }")
    print(f"Collected {len(jobs)} jobs.")

    return jobs


if __name__ == "__main__":
    MAX_QUEUE_SIZE = 15             # max concurrent jobs in SLURM queue
    POLL_INTERVAL = 3 * 60 * 60    # seconds between queue checks
    TARGET_JOB_NAME = "run_baseline_a100.sh"  # SLURM job name to track
    RUN = True                     # False = collect commands only, True = submit

    jobs_to_submit = create_commands()

    run_manager(MAX_QUEUE_SIZE, POLL_INTERVAL, TARGET_JOB_NAME, jobs_to_submit, run=RUN)
