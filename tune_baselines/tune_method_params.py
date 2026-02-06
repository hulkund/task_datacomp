import os
import subprocess

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"
RUN_CSV_BASELNE = ROOT_DIR / "run_csv_baseline.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

baselines_list = ["no_filter", "random_filter", "clip_score", "match_dist", "tsds", "gradmatch", "gradmatch_acf", "glister"]

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
    # ('AutoArborist', 'val3', 'test3'),
    # ('AutoArborist', 'val4', 'test4'),
    # ('GeoDE', 'val1', 'test1'),
    # ('GeoDE', 'val2', 'test2'),
    # ('GeoDE', 'val3', 'test3'),
    # ('GeoDE', 'val4', 'test4'),
]

supervised = "True"
use_pretrained_warmstart = "True"

with open(str(DATASETS_CONFIG), 'r') as file:
    datasets_config = yaml.safe_load(file)

### End of evaluation constants ####

total_jobs = 0
jobs_to_do = 0

for baseline in baselines_list:
    print("="*50)
    print(f"Tuning method params for {baseline}")
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params, baseline):
        for dataset, val_split, test_split in dataset_list:
            total_jobs += 1
            embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"

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
                    command = ["sbatch", str(RUN_CSV_BASELNE), baseline, dataset, task_num, fraction, subset_path, random_seed]
                else:
                    command = ["sbatch", str(RUN_BASELINE), baseline, embedding_path, subset_path, fraction, val_embedding_path, centroids_path, supervised, random_seed]
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

                    save_matrices_path = save_folder + f"deploy_{test_split[4]}_matrices/"

                    if not os.path.exists(save_matrices_path):
                        os.makedirs(save_matrices_path)

                    command.append("--save_matrices_path")
                    command.append(save_matrices_path)

                jobs_to_do += 1
                print(subset_path)
                subprocess.call(command)

print(f"{jobs_to_do = }")
print(f"{total_jobs = }")
