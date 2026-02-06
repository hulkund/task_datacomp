import os
import subprocess

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"
RUN_CSV_BASELNE = ROOT_DIR / "run_csv_baseline.sh"
DATASETS_CONFIG = ROOT_DIR / "configs/datasets.yaml"

baselines_list = ["gradmatch_acf"]

sweep_dict = create_sweep_dict()

### For evaluation ###

# Instead of using a config to define the 'tasks' we want to evaluate, we define them here
# dataset_list = [('iWildCam', 'val1', 'test1')] # (dataset, val_split, test_split)

dataset_list = [
    # ('iWildCam', 'val1', 'test1'),        # DONE (0 fails)
    # ('iWildCam', 'val2', 'test2'),        # DONE (9 fails)
    # ('iWildCam', 'val3', 'test3'),        # DONE (11 fails)
    # ('iWildCam', 'val4', 'test4'),        # DONE (9 fails)
    # ('AutoArborist', 'val1', 'test1'),
    # ('AutoArborist', 'val2', 'test2'),
    # ('AutoArborist', 'val3', 'test3'),
    # ('AutoArborist', 'val4', 'test4'),
    ('GeoDE', 'val1', 'test1'),           # DONE (5 fails)
    # ('GeoDE', 'val2', 'test2'),           # DONE (10 fails)
    # ('GeoDE', 'val3', 'test3'),           # DONE (0 fails)
    # ('GeoDE', 'val4', 'test4'),           # DOING (1 fail)
]

supervised = "True"
use_pretrained_warmstart = "True"

### End of evaluation constants ####

total_jobs = 0

for baseline in baselines_list:
    params = sweep_dict[baseline]
    for param_setting in get_sweep_combinations(params, baseline):

        for dataset, val_split, test_split in dataset_list:
            embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
            centroids_path      = f"all_datasets/{dataset}/centroids/train_centroids.pt"
            val_embedding_path  = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
            
            save_folder_sb_16 = create_save_folder(dataset, baseline, param_setting)
            subset_path_sb_16 = save_folder_sb_16 + f"{test_split}_subset.npy"
            
            if not os.path.exists(subset_path_sb_16):
                param_setting['selection_batch'] = 4 # try lower selection batch
                save_folder = create_save_folder(dataset, baseline, param_setting)
                subset_path = save_folder + f"{test_split}_subset.npy"

                if os.path.exists(subset_path):
                    print("Already exists")
                    continue

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

                total_jobs += 1
                print(subset_path)
                # subprocess.call(command)

print(f"{total_jobs = }")