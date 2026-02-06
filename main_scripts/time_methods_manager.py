import os
import subprocess

from utils import *
from manager_wrapper import run_manager

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_benchmark.sh"

BASELINES = ["gradmatch_acf", "gradmatch", "glister"]

NUM_EPOCHS = 50
MODEL_ARCH = "ResNet18"
SUPERVISED = True
USE_PRETRAINED_WARMSTART = True

DATASET_LIST = [
    ('iWildCam', 'val1', 'test1'),
    ('iWildCam', 'val2', 'test2'),
    ('iWildCam', 'val3', 'test3'),
    ('iWildCam', 'val4', 'test4'),
]

sweep_dict = create_sweep_dict()

def create_commands():
    jobs_to_do = 0

    commands = []

    for baseline in BASELINES:
        params = sweep_dict[baseline]
        for param_setting in get_sweep_combinations(params, baseline):

            for dataset, val_split, test_split in DATASET_LIST:
                embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                centroids_path      = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                val_embedding_path  = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
                save_folder = create_save_folder(dataset, baseline, param_setting)

                subset_path = save_folder + f"{test_split}_subset.npy"
                time_path = save_folder + f"{test_split}_time.txt"

                if os.path.exists(time_path):
                    continue

                assert "fraction" in param_setting
                fraction = str(param_setting["fraction"])
                random_seed = str(param_setting["random_seed"])

                command = [
                    str(RUN_BASELINE), baseline, embedding_path, subset_path,
                    fraction, val_embedding_path, centroids_path, str(SUPERVISED), random_seed,
                ]

                for k, v in param_setting.items():
                    if k == "fraction":
                        continue
                    command += [f"--{k}", str(v)]

                command += [
                    "--model", MODEL_ARCH,
                    "--num_epochs", str(NUM_EPOCHS),
                    "--use_pretrained_warmstart", str(USE_PRETRAINED_WARMSTART),
                ]

                ckpt_dir = create_warmstart_ckpt_dir(
                    dataset=dataset,
                    val_split=val_split,
                    method=baseline,
                    model=MODEL_ARCH,
                    num_epochs=NUM_EPOCHS,
                    random_seed=random_seed,
                )

                command += [
                    "--warmstart_ckpt_dir", ckpt_dir,
                    "--time_path", time_path,
                ]

                jobs_to_do += 1
                print(time_path)
                commands.append(command)

    print(f"{jobs_to_do = }")

    return commands

if __name__ == "__main__":
    MAX_QUEUE_SIZE = 20
    POLL_INTERVAL = 60*30
    TARGET_JOB_NAME = "run_benchmark.sh"

    jobs_to_submit = create_commands()

    run_manager(MAX_QUEUE_SIZE, POLL_INTERVAL, TARGET_JOB_NAME, jobs_to_submit, run=True)
