import itertools
import json
import os
import subprocess
import shlex
import yaml

# full list
# dataset_list = ['GeoDE', 'AutoArborist', 'iWildCam', 'SelfDrivingCar', 'FMoW'] #NEED CROPHARVEST & GALAXYZOO
# baselines_list = ["no_filter", "clip_score", "random_filter", "image_based", "image_alignment"]#, "match_dist", "match_label"]
# finetune_list = ["linear_probe", "full_finetune"]
# lr_list = [0.01, 0.001, 0.0001]

# tester list#NEED CROPHARVEST & GALAXYZOO
# baselines_list = ["image_alignment","no_filter","image_based","random_filter"]
# baselines_list = ["match_dist","match_label"]#,"match_dist","match_label"]

# real
# dataset_list = ['AutoArborist'] 
# baselines_list = ["no_filter", "clip_score", "random_filter", "image_based", "image_alignment", "match_dist", "match_label"]
# finetune_list = ["full_finetune"]
# lr_list = [0.01,0.001]
# batch_size_list = [32]

dataset_list = ['GeoDE'] 
baselines_list = ["zcore"]
finetune_list = ["full_finetune_resnet50"]
lr_list = [0.001]
batch_size_list = [128]

# Open the YAML baselines configuration file
with open('configs/baselines.yaml', 'r') as file:
    baselines_config = yaml.safe_load(file)

# Open the YAML datasets configuration file
with open('configs/datasets.yaml', 'r') as file:
    datasets_config = yaml.safe_load(file)

for dataset in dataset_list:
    for baseline in baselines_list:
        for finetune_type in finetune_list:
            if finetune_type=="linear_probe": lr_list = [0]
            else: lr_list = lr_list
            for lr in lr_list:
                for batch_size in batch_size_list:
                    fraction_list = baselines_config[baseline]["fraction_list"]
                    if baselines_config[baseline]["task"] == "tasks": task_list = datasets_config[dataset]["task_list"]
                    else: task_list = ["all"]
                    # embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                    # centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                    embedding_path = f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{dataset}/embeddings/train_embeddings.npy"
                    centroids_path = f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{dataset}/centroids/train_centroids.pt"
                    for fraction in fraction_list:
                        for task in task_list:
                            if task == "all" : val_embedding_path=""
                            else: val_embedding_path=f"/data/vision/beery/scratch/neha/task-datacomp/all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
                            save_folder = f"experiments_again/{dataset}/{baseline}_{fraction}/"
                            save_path= save_folder+f"{task}_subset.npy"
                            checkpoint_path = save_folder+f"{task}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
                            training_task = datasets_config[dataset]["training_task"]
                            if not os.path.exists(save_path):
                                print(save_path)
                                if baseline in ["match_label", "match_dist"]:
                                    task_num=task[4]
                                    subprocess.call(shlex.split('sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, 
                                                                                                                      dataset, 
                                                                                                                      task_num, 
                                                                                                                      fraction, 
                                                                                                                      save_path)))
    
                                else:
                                    extra_args = baselines_config[baseline].get("extra_args", "")
                                    subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s" %s'%(baseline,
                                                                                                                       embedding_path,
                                                                                                                      save_path,
                                                                                                                      fraction,
                                                                                                                      val_embedding_path,
                                                                                                                      centroids_path,
                                                                                                                      extra_args)))
                            if not os.path.exists(save_folder+f"{task}_{finetune_type}_lr={lr}_metrics.json"):
                                print(f"calling running new training job with the following args {dataset}, {training_task}")
                                subprocess.call(shlex.split('sbatch training/run_new_train.sh "%s" "%s" "%s" "%s" %s %s %s "%s" %s'%(dataset, 
                                                                                                                               save_path, 
                                                                                                                               save_folder,
                                                                                                                        'configs/datasets.yaml',
                                                                                                                               lr,
                                                                                                                               finetune_type,
                                                                                                                               batch_size,
                                                                                                                               checkpoint_path,
                                                                                                                               training_task)))