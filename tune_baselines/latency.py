import os
import subprocess

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_benchmark.sh"

baselines_list = ["gradmatch_acf", "gradmatch", "glister"]

sweep_dict = create_sweep_dict()

### For evaluation ###

# Instead of using a config to define the 'tasks' we want to evaluate, we define them here
# dataset_list = [('iWildCam', 'val1', 'test1')] # (dataset, val_split, test_split)

dataset_list = [
    ('iWildCam', 'val1', 'test1'),
    ('iWildCam', 'val2', 'test2'), # 1 failed
    ('iWildCam', 'val3', 'test3'), # 3 failed
    # ('iWildCam', 'val4', 'test4'), # 10 failed
]

supervised = "True"
use_pretrained_warmstart = "True"

### End of evaluation constants ####

def create_commands():
    total_jobs = 0
    jobs_to_do = 0

    commands = []

    for baseline in baselines_list:
        params = sweep_dict[baseline]
        for param_setting in get_sweep_combinations(params, baseline):

            for dataset, val_split, test_split in dataset_list:
                total_jobs += 1
                embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"

                centroids_path      = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                val_embedding_path  = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
                save_folder = create_save_folder(dataset, baseline, param_setting)

                subset_path = save_folder + f"{test_split}_subset.npy"
                time_path = save_folder + f"{test_split}_time.txt"

                if not os.path.exists(time_path):
                    assert "fraction" in param_setting
                    fraction = str(param_setting["fraction"])
                    random_seed = str(param_setting["random_seed"])
                    
                    command = [str(RUN_BASELINE), baseline, embedding_path, subset_path, fraction, val_embedding_path, centroids_path, supervised, random_seed]
                    
                    for k, v in param_setting.items():
                        if k == "fraction": continue
                        command.append(f"--{k}")
                        command.append(str(v))
                
                        command.append("--model")
                        command.append("ResNet18")

                        command.append("--num_epochs")
                        command.append(str(50))

                    command.append("--use_pretrained_warmstart")
                    command.append(use_pretrained_warmstart)

                    ckpt_dir = create_warmstart_ckpt_dir(
                        dataset=dataset,
                        val_split=val_split,
                        method=baseline,
                        model="ResNet18",
                        num_epochs=50,
                        random_seed=random_seed
                    )

                    ckpt_path = ckpt_dir + "warmstart_weights.pth"

                    command.append("--warmstart_ckpt_dir")
                    command.append(ckpt_dir)

                    command.append("--time_path")
                    command.append(time_path)

                    jobs_to_do += 1
                    print(time_path)
                    commands.append(command)
                    # subprocess.call(["sbatch"] + command)

    print(f"{jobs_to_do = }")
    print(f"{total_jobs = }")

    return commands

from manager_wrapper import run_manager

if __name__ == "__main__":
    MAX_QUEUE_SIZE = 20
    POLL_INTERVAL = 60*30
    TARGET_JOB_NAME = "run_latency.sh"

    jobs_to_submit = create_commands()

    run_manager(MAX_QUEUE_SIZE, POLL_INTERVAL, TARGET_JOB_NAME, jobs_to_submit, run=True)

