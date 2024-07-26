import itertools
import json
import os
import subprocess
import shlex

with open('config/baselines.yaml', 'r') as file:
    baselines_config = yaml.safe_load(file)

image_dataset_list = ['COOS','FMoW','iWildCam']

for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
    for baseline in baselines_config['baselines']:
        for fraction in baseline['fraction_list']:
            for task in baseline['task_list']:
                if task=='all':
                    save_folder = f"experiments/{dataset}/{baseline_fraction}/"
                    save_path= save_folder+f"subset.npy"
                    if not os.path.exists(save_path):
                        subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, embedding_path, save_path, fraction)))
                    if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
                        subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))
                else:
                    val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
                    save_folder = f"experiments/{dataset}/{baseline_fraction}/"
                    save_path= save_folder+f"{task}_subset.npy"
                    if not os.path.exists(save_path):
                        subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'%(baseline, embedding_path, save_path, fraction, val_embedding_path, centroids_path)))
                    if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
                        subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))
