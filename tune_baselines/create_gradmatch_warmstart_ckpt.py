import os
import subprocess

from utils import *

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

RUN_BASELINE = ROOT_DIR / "run_baseline.sh"

baselines_list = ["gradmatch"]

### For evaluation ###

# [(dataset_name, val_split, num_epochs, model_architecture, random_seed), ...]
val1_param_combinations = [
    ("iWildCam", "val1", 50, "ResNet18", 0),
    ("iWildCam", "val1", 50, "ResNet18", 1),
    ("iWildCam", "val1", 50, "ResNet18", 42),
    ("AutoArborist", "val1", 50, "ResNet18", 0),
    ("AutoArborist", "val1", 50, "ResNet18", 1),
    ("AutoArborist", "val1", 50, "ResNet18", 42),
    ("GeoDE", "val1", 50, "ResNet18", 0),
    ("GeoDE", "val1", 50, "ResNet18", 1),
    ("GeoDE", "val1", 50, "ResNet18", 42),
]

val2_param_combinations = [
    ("iWildCam", "val2", 50, "ResNet18", 0),
    ("iWildCam", "val2", 50, "ResNet18", 1),
    ("iWildCam", "val2", 50, "ResNet18", 42),
    ("AutoArborist", "val2", 50, "ResNet18", 0),
    ("AutoArborist", "val2", 50, "ResNet18", 1),
    ("AutoArborist", "val2", 50, "ResNet18", 42),
    ("GeoDE", "val2", 50, "ResNet18", 0),
    ("GeoDE", "val2", 50, "ResNet18", 1),
    ("GeoDE", "val2", 50, "ResNet18", 42),
]

val3_param_combinations = [
    ("iWildCam", "val3", 50, "ResNet18", 0),
    ("iWildCam", "val3", 50, "ResNet18", 1),
    ("iWildCam", "val3", 50, "ResNet18", 42),
    ("AutoArborist", "val3", 50, "ResNet18", 0),
    ("AutoArborist", "val3", 50, "ResNet18", 1),
    ("AutoArborist", "val3", 50, "ResNet18", 42),
    ("GeoDE", "val3", 50, "ResNet18", 0),
    ("GeoDE", "val3", 50, "ResNet18", 1),
    ("GeoDE", "val3", 50, "ResNet18", 42),
]

val4_param_combinations = [
    ("iWildCam", "val4", 50, "ResNet18", 0),
    ("iWildCam", "val4", 50, "ResNet18", 1),
    ("iWildCam", "val4", 50, "ResNet18", 42),
    ("AutoArborist", "val4", 50, "ResNet18", 0),             #DOING
    ("AutoArborist", "val4", 50, "ResNet18", 1),             #DOING
    ("AutoArborist", "val4", 50, "ResNet18", 42),            #DONE
    ("GeoDE", "val4", 50, "ResNet18", 0),
    ("GeoDE", "val4", 50, "ResNet18", 1),
    ("GeoDE", "val4", 50, "ResNet18", 42),
]

param_combinations = val1_param_combinations + val2_param_combinations + val3_param_combinations + val4_param_combinations

supervised = "True"
use_pretrained_warmstart = "False"

### Hardcoded args for legacy ###
selection_batch = "16"
selection_lr = "0.01"
lam = "0.5"

### End of evaluation constants ####

total_jobs = 0

for baseline in baselines_list:
    print("="*50)
    for param_setting in param_combinations:
        print("Trying param configuration:", param_setting)
        dataset, val_split, num_epochs, model_arch, random_seed = param_setting
        
        embedding_path      = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
        centroids_path      = f"all_datasets/{dataset}/centroids/train_centroids.pt"
        
        val_embedding_path  = f"all_datasets/{dataset}/embeddings/{val_split}_embeddings.npy"
        
        ckpt_dir = create_warmstart_ckpt_dir(
            dataset=dataset,
            val_split=val_split,
            method=baseline,
            model=model_arch,
            num_epochs=num_epochs,
            random_seed=random_seed
        )

        ckpt_path = ckpt_dir + "warmstart_weights.pth"
        if use_pretrained_warmstart == "True" or not os.path.exists(ckpt_path):
            fraction = "1"
            random_seed = str(random_seed)

            command = ["sbatch", str(RUN_BASELINE), baseline, embedding_path, ckpt_path, fraction, val_embedding_path, centroids_path, supervised, random_seed]
            
            command.append("--selection_batch")
            command.append(selection_batch)
            command.append("--selection_lr")
            command.append(selection_lr)
            command.append("--lam")
            command.append(lam)

            command.append("--model")
            command.append(model_arch)

            command.append("--num_epochs")
            command.append(str(num_epochs))

            command.append("--use_pretrained_warmstart")
            command.append(use_pretrained_warmstart)
            
            command.append("--warmstart_ckpt_dir")
            command.append(ckpt_dir)
                
            total_jobs += 1
            subprocess.call(command)

print(f"{total_jobs = }")