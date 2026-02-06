import os
import subprocess
import time
import yaml
from pathlib import Path

# --- QUEUE CONFIGURATION ---
MAX_QUEUE_SIZE = 15
POLL_INTERVAL = 3*60*60
USER_NAME = "evelynz"
# TARGET_JOB_NAME = "run_baseline_optimized.sh"
# TARGET_JOB_NAME = "run_baseline_gm_aa1.sh"
# TARGET_JOB_NAME = "run_baseline_beery_a100.sh"
TARGET_JOB_NAME = "run_baseline_a100.sh"
RUN = True
# RUN = False

def get_specific_job_count():
    """Counts only jobs matching the specific name."""
    try:
        # -n filters by job name, -u by user, -h removes header
        cmd = ["squeue", "-u", USER_NAME, "-n", TARGET_JOB_NAME, "-h", "-t", "PD,R"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = [line for line in result.stdout.split('\n') if line.strip()]
        return len(lines)
    except Exception as e:
        print(f"Error checking queue: {e}")
        return MAX_QUEUE_SIZE

from utils import *

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"
RUN_CSV_BASELNE = ROOT_DIR / "run_csv_baseline.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

# baselines_list = ["no_filter", "random_filter", "clip_score", "match_dist", "tsds", "gradmatch", "gradmatch_acf", "glister"]
baselines_list = ["gradmatch", "gradmatch_acf"]

sweep_dict = create_sweep_dict()

### For evaluation ###

# Instead of using a config to define the 'tasks' we want to evaluate, we define them here
# dataset_list = [('iWildCam', 'val1', 'test1')] # (dataset, val_split, test_split)

dataset_list = [
    # ('iWildCam', 'val1', 'test1'),
    # ('iWildCam', 'val2', 'test2'),
    # ('iWildCam', 'val3', 'test3'),
    # ('iWildCam', 'val4', 'test4'),
    # ('AutoArborist', 'val1', 'test1'),
    # ('AutoArborist', 'val2', 'test2'),
    ('AutoArborist', 'val3', 'test3'),
    # ('AutoArborist', 'val4', 'test4'),
    # ('GeoDE', 'val1', 'test1'),
    # ('GeoDE', 'val2', 'test2'),
    # ('GeoDE', 'val3', 'test3'),
    # ('GeoDE', 'val4', 'test4'),
]

# dataset_list = [
#     # ('iWildCam', 'val1', 'test1'),        # DONE (0 fails)
#     # ('iWildCam', 'val2', 'test2'),        # DONE (9 fails)
#     # ('iWildCam', 'val3', 'test3'),        # DONE (11 fails)
#     # ('iWildCam', 'val4', 'test4'),        # DONE (9 fails)
#     # ('AutoArborist', 'val1', 'test1'),
#     # ('AutoArborist', 'val2', 'test2'),
#     # ('AutoArborist', 'val3', 'test3'),
#     # ('AutoArborist', 'val4', 'test4'),
#     # ('GeoDE', 'val1', 'test1'),           # DONE (5 fails)
#     # ('GeoDE', 'val2', 'test2'),           # DONE (10 fails)
#     # ('GeoDE', 'val3', 'test3'),           # DONE (0 fails)
#     # ('GeoDE', 'val4', 'test4'),           # DOING (1 fail)
# ]

supervised = "True"
# supervised = "False"

use_pretrained_warmstart = "True"

with open(str(DATASETS_CONFIG), 'r') as file:
    datasets_config = yaml.safe_load(file)

### End of evaluation constants ####

jobs_to_submit = []

for baseline in baselines_list:
    print("="*50)
    print(f"Tuning method params for {baseline}")
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params, baseline):
        print("Trying param configuration:", param_setting)

        for dataset, val_split, test_split in dataset_list:
            embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
            
            # hard-coded case
            if baseline == "zcore":
                embedding_path = "/data/vision/beery/scratch/neha/task-datacomp/experiments_again/iWildCam/no_filter_1/embeddings/all_subset_resnet50.npy"

            centroids_path      = f"all_datasets/{dataset}/centroids/train_centroids.pt"
            val_embedding_path  = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
            save_folder = create_save_folder(dataset, baseline, param_setting)

            # Create subset
            subset_path = save_folder + f"{test_split}_subset.npy"
            if not os.path.exists(subset_path):
                assert "fraction" in param_setting
                fraction = str(param_setting["fraction"])
                random_seed = str(param_setting["random_seed"])
                
                if  baseline in ["match_dist", "match_label"]:
                    task_num = test_split[4]
                    command = [str(RUN_CSV_BASELNE), baseline, dataset, task_num, fraction, subset_path, random_seed]
                else:
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

                    ckpt_path = ckpt_dir + "warmstart_weights.pth"

                    command.append("--warmstart_ckpt_dir")
                    command.append(ckpt_dir)

                print("Running command to create subset:", " ".join(command))
                jobs_to_submit.append(command)


# --- SUBMISSION MONITOR ---
print(f"{baselines_list = }")
print(f"{dataset_list = }")
print(f"Collected {len(jobs_to_submit)} jobs. Starting managed submission...")

submitted = 0
total = len(jobs_to_submit)

while jobs_to_submit:
    current_active = get_specific_job_count()
    
    if current_active < MAX_QUEUE_SIZE:
        num_to_spawn = MAX_QUEUE_SIZE - current_active
        for _ in range(num_to_spawn):
            if not jobs_to_submit:
                break
            
            cmd = jobs_to_submit.pop(0)
            command = ["sbatch", "--job-name", TARGET_JOB_NAME] + cmd
            
            if RUN == True:
                subprocess.run(command)
            
            submitted += 1
            print(f"[{submitted}/{total}] Submitted. '{TARGET_JOB_NAME}' active: {current_active + 1}")
            current_active += 1
    else:
        print(f"Queue full for '{TARGET_JOB_NAME}' ({current_active}/{MAX_QUEUE_SIZE}). Waiting...")
        time.sleep(POLL_INTERVAL)

print("All baseline jobs submitted.")