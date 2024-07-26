import itertools
import json
import os
import subprocess
import shlex

# Define your lists of parameters
image_dataset_list = ['COOS','FMoW','iWildCam']
baselines_list = ["no_filter"]
                #"basic_filter"]
fraction_baselines_list = ["clip_score",
                "random_filter"] 
image_baselines_list = ["image_based"]#,"image_based_intersect_clip_score"]
tasks = ["test1", "test2", "test3", "test4"]
fraction_list = [0.05, 0.10, 0.25, 0.50, 0.75,0.9,0.95, 0.99]
# C_list=[0.1,0.25,0.5,0.75]


for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
    for baseline in ["image_based"]:
        for task in tasks:
            val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
            save_folder = f"experiments/{dataset}/{baseline}/"
            save_path= save_folder+f"{task}_subset.npy"
            if not os.path.exists(save_path):
                subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'%(baseline, embedding_path, save_path, 1, val_embedding_path, centroids_path)))
            if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
                subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

# image_clip
for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
    for baseline in ['image_clip','image_alignment']:
        for task in tasks:
            val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
            for fraction in fraction_list:
                save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
                save_path= save_folder+f"{task}_subset.npy"
                if not os.path.exists(save_path):
                    subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'%(baseline, embedding_path, save_path, fraction, val_embedding_path, centroids_path)))
                if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
                    subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))
        
# baselines_list    
for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    for baseline in baselines_list:
         save_folder = f"experiments/{dataset}/{baseline}/"
         save_path= save_folder+"subset.npy"
         if not os.path.exists(save_path):
             subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, 1)))
         if not os.path.exists(save_folder+"test1_C=0.75_metrics.json"):
             subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

#fraction_baselines list
for dataset in image_dataset_list:
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    for baseline in fraction_baselines_list:
        for fraction in fraction_list:
             save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
             save_path= save_folder+"subset.npy"
             if not os.path.exists(save_path):
                 subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s'%(baseline, embedding_path, save_path, fraction)))
             if not os.path.exists(save_folder+"test1_C=0.75_metrics.json"):
                 subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

#match_dist
for dataset in image_dataset_list:
    baseline = "match_dist"
    for fraction in fraction_list:
        for task in tasks:
            task_num=int(task[4])
            save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
            save_path= save_folder+f"{task}_subset.npy"
            if not os.path.exists(save_path):
                subprocess.call(shlex.split('sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, dataset, task_num, fraction, save_path)))
            if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
                subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

# match_label
for dataset in ['iWildCam','FMoW']:
    baseline = "match_label"
    for task in tasks:
        task_num=int(task[4])
        save_folder = f"experiments/{dataset}/{baseline}/"
        save_path= save_folder+f"{task}_subset.npy"
        if not os.path.exists(save_path):
            subprocess.call(shlex.split('sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, dataset, task_num, 1, save_path)))
        if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
            subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))


text_baselines_dict={'random_filter':fraction_list,
                     'no_filter':[1]}
                    # 'text_based'}


# ## TEXT BASED BASELINES ##
# for dataset in ['CivilComments']:
#     embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
#     for baseline in text_baselines_dict:
#         for fraction in text_baselines_dict[baseline]:
#             save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
#             save_path= save_folder+f"subset.npy"
#             if not os.path.exists(save_path):
#                 subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s"'%(baseline, embedding_path, save_path, fraction)))
#             if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
#                 subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

# text_alignment_baselines_dict={'text_alignment':fraction_list,
#                      'text_based':[1]}
# for dataset in ['CivilComments']:
#     embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
#     centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
#     for baseline in text_alignment_baselines_dict:
#         for fraction in text_alignment_baselines_dict[baseline]:
#             for task in tasks:
#                 task_num=int(task[4])
#                 save_folder = f"experiments/{dataset}/{baseline}_{fraction}/"
#                 save_path= save_folder+f"{task}_subset.npy"
#                 val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
#                 if not os.path.exists(save_path):
#                     subprocess.call(shlex.split('sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'%(baseline, embedding_path, save_path, fraction, val_embedding_path,centroids_path)))
#                 if not os.path.exists(save_folder+f"test{task[4]}_C=0.75_metrics.json"):
#                     subprocess.call(shlex.split('sbatch baselines/run_clip_linear_probe.sh "%s" "%s" "%s"'%(dataset, save_path, save_folder)))

# --name "$1" \
# 	--embedding_path "$2" \
# 	--save_path "$3" \
# 	--fraction $4 \
#     --val_embedding_path "$5" \
#     --centroids_path "$6" \
        
    
    
        
    



