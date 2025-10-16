import os
import yaml
from typing import Iterator
from itertools import product

ROOT_DIR = "/data/vision/beery/scratch/evelyn/task_datacomp"
config_path = f"{ROOT_DIR}/tune_baselines/config.yaml"

def create_sweep_dict() -> dict:
    with open(config_path, "r") as file:
        sweep_config = yaml.safe_load(file)
    return sweep_config

def get_sweep_combinations(params: dict) -> Iterator:
    keys = list(params.keys())
    values = list(params.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

def create_save_folder(dataset: str, method: str, param_setting: dict) -> str:
    param_str = "_".join(f"{k}_{v}" for k, v in sorted(param_setting.items()))
    save_folder = f"{ROOT_DIR}/experiments/{dataset}/{method}_{param_str}/"
    
    if not os.path.exists(save_folder):
        print("Creating experiment folder:", save_folder)
        os.makedirs(save_folder)

    return save_folder