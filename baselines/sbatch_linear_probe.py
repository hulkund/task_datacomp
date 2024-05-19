import itertools
import json
import os
import subprocess
import shlex

# Define your lists of parameters
dataset_list = ['COOS', 'FMoW']
baselines_list = ["no_filter",]
                #"basic_filter"]
                #"image_based",]
fraction_baselines_list = [
                #"image_based_intersect_clip_score",
                "clip_score",
                "random_filter"]    
# tasks = ["task1", "task2", "task2", "task4"]
fraction_list = [0.05, 0.10, 0.25, 0.50, 0.75]

# Generate config files
for dataset in dataset_list:
    for baseline in baselines_list:
        subset_path = f"experiments/{dataset}/{baseline}/subset.npy"
        if not os.path.exists(save_path):
            subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, None)))
        
    for baseline in fraction_baselines_list:
        for fraction in fraction_list:
            subset_path = f"experiments/{dataset}/{baseline}_{fraction}/subset.npy"
            subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, fraction)))





