import os
import subprocess

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"
RUN_CSV_BASELNE = ROOT_DIR / "run_csv_baseline.sh"
RUN_NEW_TRAIN = ROOT_DIR / "baselines/run_new_train.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

# baselines_list = ["gradmatch"]
# baselines_list = ["no_filter", "random_filter", "match_dist"]
baselines_list = ["zcore"]

sweep_dict = create_sweep_dict()

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
    print("="*50)
    print(f"Tuning method params for {baseline}")
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params):
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

                        # Create subset
                        subset_path = save_folder + f"{test_split}_subset.npy"
                        if not os.path.exists(subset_path):
                            assert "fraction" in param_setting
                            fraction = str(param_setting["fraction"])
                            
                            if  baseline in ["match_dist", "match_label"]:
                                task_num = test_split[4]
                                command = ["sbatch", str(RUN_CSV_BASELNE), baseline, dataset, task_num, fraction, subset_path]
                            else:
                                command = ["sbatch", str(RUN_BASELINE), baseline, embedding_path, subset_path, fraction, val_embedding_path, centroids_path]
                                for k, v in param_setting.items():
                                    if k == "fraction": continue
                                    command.append(f"--{k}")
                                    command.append(str(v))
                            
                            print("Running command to create subset:", " ".join(command))
                            subprocess.call(command)

                        # # Run new training job to evaluate the param settings
                        # metrics_path    = save_folder + f"{test_split}_{finetune_type}_lr={lr}_metrics.json"
                        # checkpoint_path = save_folder + f"{test_split}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
                        # training_task = datasets_config[dataset]["training_task"]
                        # if not os.path.exists(metrics_path) and os.path.exists(subset_path):
                        #     command = ["sbatch", str(RUN_NEW_TRAIN), dataset, subset_path, save_folder, "configs/datasets.yaml", str(lr), finetune_type, str(batch_size), checkpoint_path, training_task]
                        #     print("Running command to run new train:", " ".join(command))
                        #     subprocess.call(command)

