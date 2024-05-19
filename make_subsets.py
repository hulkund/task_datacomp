import itertools
import json
import os
import subprocess
import shlex

# Define your lists of parameters
dataset_list = ['FMoW','COOS']
# baselines_list=[]
baselines_list = ["no_filter",
                #"basic_filter"]
                "image_based",]
fraction_baselines_list = [
                #"image_based_intersect_clip_score",
                "clip_score",
                "random_filter"]   
task_based_baselines = 
# tasks = ["task1", "task2", "task2", "task4"]
fraction_list = [0.05, 0.10, 0.25, 0.50, 0.75]
C_list=[0.1,0.25,0.5,0.75]

# Generate config files
for dataset in dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    for baseline in baselines_list:
        save_folder = f"experiments/{dataset}/{baseline}/"
        save_path= save_folder+"subset.npy"
        if not os.path.exists(save_path):
            subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, 1)))
        else:
            subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))
        
    for baseline in fraction_baselines_list:
        for fraction in fraction_list:
            save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
            save_path= save_folder+"subset.npy"
            if not os.path.exists(save_path):
                subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, fraction)))
            else:
                subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))






