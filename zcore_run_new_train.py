import yaml
import os
import subprocess

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_NEW_TRAIN = ROOT_DIR / "task_datacomp/baselines/run_new_train.sh"
DATASETS_CONFIG = ROOT_DIR / "task_datacomp/configs/datasets.yaml"

# baselines_list = ["gradmatch"]
# baselines_list = ["no_filter", "random_filter", "match_dist"]
baselines_list = ["zcore"]
fractions_list = [0.25]

### For evaluation ###

# Instead of using a config to define the 'tasks' we want to evaluate, we define them here
dataset_list = [('iWildCam', 'val1', 'test1')] # (dataset, val_split, test_split)

finetune_list = ["full_finetune_resnet50"]
lr_list = [0.001]
batch_size_list = [128]

with open(str(DATASETS_CONFIG), 'r') as file:
    datasets_config = yaml.safe_load(file)

### End of evaluation constants ####

for baseline in baselines_list:
    for fraction in fractions_list:
        for dataset, val_split, test_split in dataset_list:
            for finetune_type in finetune_list:
                if finetune_type=="linear_probe": lr_list = [0]
                else: lr_list = lr_list
                for lr in lr_list:
                    for batch_size in batch_size_list:
                        embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                        save_folder = f"/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/{baseline}_fraction_{fraction}/"
                        subset_path = save_folder + f"{test_split}_subset.npy"
                        
                        # Run new training job to evaluate the param settings
                        metrics_path    = save_folder + f"{test_split}_{finetune_type}_lr={lr}_metrics.json"
                        checkpoint_path = save_folder + f"{test_split}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
                        training_task = datasets_config[dataset]["training_task"]
                        if not os.path.exists(metrics_path) and os.path.exists(subset_path):
                            command = ["sbatch", str(RUN_NEW_TRAIN), dataset, subset_path, save_folder, "configs/datasets.yaml", str(lr), finetune_type, str(batch_size), checkpoint_path, training_task]
                            print("Running command to run new train:", " ".join(command))
                            subprocess.call(command)

