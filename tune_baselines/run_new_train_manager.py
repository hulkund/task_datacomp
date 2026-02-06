import time
import os
import subprocess

from utils import *

from pathlib import Path

# --- NEW CONFIGURATION ---
MAX_QUEUE_SIZE = 10
POLL_INTERVAL = 5*60  # Check every 10 minutes
USER_NAME = "evelynz"
TARGET_JOB_NAME = "run_new_train.sh"

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_NEW_TRAIN = ROOT_DIR / "baselines/run_new_train.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

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

# baselines_list = ["no_filter", "random_filter", "clip_score", "match_dist", "gradmatch_acf", "gradmatch", "glister"]
baselines_list = ["no_filter", "random_filter", "clip_score", "match_dist", "gradmatch_acf", "gradmatch", "glister", "tsds"]
# baselines_list = ["gradmatch_acf", "gradmatch"]

sweep_dict = create_sweep_dict()

### For evaluation ###

# Instead of using a config to define the 'tasks' we want to evaluate, we define them here
# dataset_list = [('iWildCam', 'val1', 'test1')] # (dataset, val_split, test_split)

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


# finetune_list = ["full_finetune_resnet50"]
finetune_list = ["lora_finetune_vit", "full_finetune_resnet50"]
lr_list = [0.001]
batch_size_list = [128]

with open(str(DATASETS_CONFIG), 'r') as file:
    datasets_config = yaml.safe_load(file)

### End of evaluation constants ####


jobs_to_submit = []
total_num_jobs = 0
total_subsets_not_created = 0

for baseline in baselines_list:
    print("="*50)
    print(f"Tuning method params for {baseline}")
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params, baseline):
        print("Trying param configuration:", param_setting)

        for dataset, val_split, test_split in dataset_list:
            for finetune_type in finetune_list:
                if finetune_type=="linear_probe": lr_list = [0]
                else: lr_list = lr_list
                for lr in lr_list:
                    for batch_size in batch_size_list:
                        embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                        
                        # hard-coded case
                        if baseline == "zcore":
                            embedding_path = "/data/vision/beery/scratch/neha/task-datacomp/experiments_again/iWildCam/no_filter_1/embeddings/all_subset_resnet50.npy"

                        centroids_path      = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                        val_embedding_path  = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
                        save_folder = create_save_folder(dataset, baseline, param_setting)

                        # If subset was made, it would be in this paths
                        subset_path = save_folder + f"{test_split}_subset.npy"

                        # Run new training job to evaluate the param settings
                        metrics_path    = save_folder + f"{test_split}_{finetune_type}_lr={lr}_batchsize={batch_size}_metrics.json"
                        checkpoint_path = save_folder + f"{test_split}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
                        training_task = datasets_config[dataset]["training_task"]
                        
                        if not os.path.exists(subset_path):
                            total_subsets_not_created += 1

                        if not os.path.exists(metrics_path) and os.path.exists(subset_path):
                            command = [str(RUN_NEW_TRAIN), dataset, subset_path, save_folder, "configs/datasets.yaml", str(lr), finetune_type, str(batch_size), checkpoint_path, training_task]
                            jobs_to_submit.append(command)
                            print("Added command to jobs_to_submit:", " ".join(command))

print(f"{baselines_list = }")
print(f"{dataset_list = }")
print(f"{finetune_list = }")
print(f"total_jobs = {len(jobs_to_submit)}")
print(f"number of subsets not created: {total_subsets_not_created // len(finetune_list)}")

# --- SUBMISSION LOOP ---
submitted_count = 0
total_to_submit = len(jobs_to_submit)

while jobs_to_submit:
    current_active = get_specific_job_count()
    
    if current_active < MAX_QUEUE_SIZE:
        # Calculate how many we can submit right now
        available_slots = MAX_QUEUE_SIZE - current_active
        
        for _ in range(available_slots):
            if not jobs_to_submit:
                break
                
            command = jobs_to_submit.pop(0)
            command = ["sbatch", "--job-name", TARGET_JOB_NAME] + command
            try:
                # subprocess.run(command)
                submitted_count += 1
                print(f"[{submitted_count}/{total_to_submit}] Submitted job. Active in queue: {current_active + 1}")
                current_active += 1 # Update local count to avoid burst-overfilling
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit: {' '.join(command)}")
                
    else:
        print(f"Queue full ({current_active}/{MAX_QUEUE_SIZE}). Sleeping {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

print("Finished submitting all jobs.")
