import os
import subprocess
import shlex
import yaml


def main():
    # Configuration variables
    dataset_list = ["iWildCam"]
    baselines_list = ["no_filter"]
    finetune_list = ["full_finetune_resnet50"]
    lr_list = [0.001]
    batch_size_list = [128]
    num_epochs = 1 # just for testing purposes - set to 100 for actual runs
    baselines_config = "configs/subset_baselines.yaml"
    datasets_config_path = "configs/datasets.yaml"
    experiments_dir = "experiments_again"
    wandb_project = "DataS3 logging"
    wandb_entity = "datas3"
    wandb_group = None

    with open(baselines_config, 'r') as file:
        baselines_config = yaml.safe_load(file)

    with open(datasets_config_path, 'r') as file:
        datasets_config = yaml.safe_load(file)

    for dataset in dataset_list:
        for baseline in baselines_list:
            for finetune_type in finetune_list:
                lr_list = [0] if finetune_type == "linear_probe" else lr_list
                for lr in lr_list:
                    for batch_size in batch_size_list:
                        fraction_list = baselines_config[baseline]["fraction_list"]
                        if baselines_config[baseline]["task"] == "tasks":
                            task_list = datasets_config[dataset]["task_list"]
                        else:
                            task_list = ["all"]
                        embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
                        centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"
                        for fraction in fraction_list:
                            for task in task_list:
                                if task == "all":
                                    val_embedding_path = ""
                                else:
                                    val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"
                                save_folder = f"{experiments_dir}/{dataset}/{baseline}_{fraction}/"
                                save_path = save_folder + f"{task}_subset.npy"
                                checkpoint_path = save_folder + f"{task}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
                                training_task = datasets_config[dataset]["training_task"]

                                # Submit baseline filtering job if subset doesn't exist
                                if not os.path.exists(save_path):
                                    print(save_path)
                                    if baseline in ["match_label", "match_dist"]:
                                        task_num = task[4]
                                        subprocess.call(shlex.split(
                                            'sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"' % (
                                                baseline, dataset, task_num, fraction, save_path)))
                                    else:
                                        subprocess.call(shlex.split(
                                            'sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"' % (
                                                baseline, embedding_path, save_path, fraction,
                                                val_embedding_path, centroids_path)))

                                # Submit training job if metrics don't exist
                                if not os.path.exists(save_folder + f"{task}_{finetune_type}_lr={lr}_metrics.json"):
                                    print(f"calling running new training job with the following args {dataset}, {training_task}")
                                    wandb_project_arg = wandb_project or ""
                                    wandb_entity_arg = wandb_entity or ""
                                    wandb_group_arg = wandb_group or f"{dataset}/{baseline}"
                                    wandb_run_name = f"{baseline}/{finetune_type}/frac={fraction}/lr={lr}/{task}"
                                    subprocess.call(shlex.split(
                                        'sbatch training/run_new_train.sh "%s" "%s" "%s" "%s" %s %s %s "%s" %s "%s" "%s" "%s" "%s" %s' % (
                                            dataset, save_path, save_folder,
                                            datasets_config_path, lr, finetune_type,
                                            batch_size, checkpoint_path, training_task,
                                            wandb_project_arg, wandb_entity_arg,
                                            wandb_group_arg, wandb_run_name,
                                            num_epochs)))


if __name__ == "__main__":
    # base command
    main()
