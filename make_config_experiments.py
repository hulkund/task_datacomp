import argparse
import os
import subprocess
import shlex
import yaml
from itertools import product


def build_experiment(dataset, baseline, finetune_type, lr, batch_size,
                     fraction, task, baselines_config, datasets_config,
                     experiments_dir, datasets_config_path,
                     wandb_project="", wandb_entity="", wandb_group=None,
                     num_epochs=100, seed=42):
    """Build paths and commands for a single (filter, train) experiment."""
    embedding_path = f"all_datasets/{dataset}/embeddings/train_embeddings.npy"
    centroids_path = f"all_datasets/{dataset}/centroids/train_centroids.pt"

    if task == "all":
        val_embedding_path = ""
    else:
        val_embedding_path = f"all_datasets/{dataset}/embeddings/val{task[4]}_embeddings.npy"

    save_folder = f"{experiments_dir}/{dataset}/{baseline}_{fraction}/"
    save_path = save_folder + f"{task}_subset.npy"
    checkpoint_path = save_folder + f"{task}_finetune={finetune_type}_lr={lr}_batchsize={batch_size}"
    training_task = datasets_config[dataset]["training_task"]
    metrics_path = save_folder + f"{task}_{finetune_type}_lr={lr}_metrics.json"

    # Build filter command
    if baseline in ["match_label", "match_dist"]:
        task_num = task[4]
        filter_cmd = (
            'sbatch run_csv_baseline.sh "%s" "%s" "%s" %s "%s"'
            % (baseline, dataset, task_num, fraction, save_path)
        )
    else:
        filter_cmd = (
            'sbatch run_baseline.sh "%s" "%s" "%s" %s "%s" "%s"'
            % (baseline, embedding_path, save_path, fraction,
               val_embedding_path, centroids_path)
        )

    # Build train command
    group = wandb_group or f"{dataset}/{baseline}"
    run_name = f"{baseline}/{finetune_type}/frac={fraction}/lr={lr}/{task}"
    train_cmd = (
        'sbatch training/run_new_train.sh "%s" "%s" "%s" "%s" %s %s %s "%s" %s "%s" "%s" "%s" "%s" %s'
        % (dataset, save_path, save_folder, datasets_config_path,
           lr, finetune_type, batch_size, checkpoint_path, training_task,
           wandb_project, wandb_entity, group, run_name, num_epochs, seed)
    )

    return {
        "dataset": dataset,
        "baseline": baseline,
        "finetune_type": finetune_type,
        "lr": lr,
        "batch_size": batch_size,
        "fraction": fraction,
        "task": task,
        "save_path": save_path,
        "metrics_path": metrics_path,
        "filter_cmd": filter_cmd,
        "train_cmd": train_cmd,
    }


def generate_experiments(sweep, baselines_config, datasets_config,
                         experiments_dir, datasets_config_path,
                         wandb_project="", wandb_entity="", wandb_group=None,
                         num_epochs=100, seed=42):
    """Yield experiment dicts from a single sweep definition."""
    datasets = sweep["datasets"]
    baselines = sweep["baselines"]
    finetune_types = sweep["finetune_types"]
    lr_list = sweep.get("lr_list", [0.001])
    batch_sizes = sweep.get("batch_sizes", [128])
    num_epochs = sweep.get("num_epochs", num_epochs)
    seed = sweep.get("seed", seed)

    for dataset, baseline, finetune_type, batch_size in product(
        datasets, baselines, finetune_types, batch_sizes
    ):
        # Linear probe ignores LR
        lrs = [0] if finetune_type == "linear_probe" else lr_list

        fraction_list = baselines_config[baseline]["fraction_list"]
        if baselines_config[baseline]["task"] == "tasks":
            task_list = datasets_config[dataset]["task_list"]
        else:
            task_list = ["all"]

        for lr, fraction, task in product(lrs, fraction_list, task_list):
            yield build_experiment(
                dataset, baseline, finetune_type, lr, batch_size,
                fraction, task, baselines_config, datasets_config,
                experiments_dir, datasets_config_path,
                wandb_project, wandb_entity, wandb_group,
                num_epochs, seed,
            )


def submit_jobs(experiments, dry_run=False, filter_only=False,
                train_only=False, max_jobs=None):
    """Submit filter and/or training SLURM jobs for each experiment."""
    submitted = 0
    skipped_filter = 0
    skipped_train = 0
    skipped_no_subset = 0

    for exp in experiments:
        if max_jobs is not None and submitted >= max_jobs:
            print(f"Reached --max-jobs limit ({max_jobs}), stopping.")
            break

        # Filter job
        if not train_only:
            if os.path.exists(exp["save_path"]):
                skipped_filter += 1
            elif dry_run:
                print(f"[DRY RUN] filter: {exp['filter_cmd']}")
                submitted += 1
            else:
                print(f"Submitting filter: {exp['save_path']}")
                subprocess.call(shlex.split(exp["filter_cmd"]))
                submitted += 1

        # Training job (only if subset .npy exists)
        if not filter_only:
            if os.path.exists(exp["metrics_path"]):
                skipped_train += 1
            elif not os.path.exists(exp["save_path"]):
                skipped_no_subset += 1
            elif dry_run:
                print(f"[DRY RUN] train:  {exp['train_cmd']}")
                submitted += 1
            else:
                print(f"Submitting train: {exp['dataset']} / {exp['baseline']} / "
                      f"ft={exp['finetune_type']} / lr={exp['lr']} / "
                      f"frac={exp['fraction']} / task={exp['task']}")
                subprocess.call(shlex.split(exp["train_cmd"]))
                submitted += 1

    print(f"\nDone. Submitted: {submitted} | "
          f"Skipped (filter exists): {skipped_filter} | "
          f"Skipped (metrics exist): {skipped_train} | "
          f"Skipped (subset not ready): {skipped_no_subset}")


def main():
    parser = argparse.ArgumentParser(
        description="Run subset selection and training experiments from a config file or CLI args."
    )

    # Config-file mode
    parser.add_argument("--experiment_config", type=str, default=None,
                        help="YAML file defining experiment sweeps")

    # CLI fallback mode (same interface as make_subsets_with_config.py)
    parser.add_argument("--dataset_list", nargs="+", default=None,
                        help="Datasets to run (CLI fallback mode)")
    parser.add_argument("--baselines_list", nargs="+", default=None,
                        help="Baseline methods to run (CLI fallback mode)")
    parser.add_argument("--finetune_list", nargs="+", default=None,
                        help="Finetune types (CLI fallback mode)")
    parser.add_argument("--lr_list", nargs="+", type=float, default=[0.001, 0.0001],
                        help="Learning rates (CLI fallback mode)")
    parser.add_argument("--batch_size_list", nargs="+", type=int, default=[128],
                        help="Batch sizes (CLI fallback mode)")
    parser.add_argument("--num_epochs", type=int, required=False, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--seed", type=int, required=False, default=42,
                        help="Random seed passed to training (default: 42)")

    # Shared options
    parser.add_argument("--baselines_config", type=str, default="configs/subset_baselines.yaml")
    parser.add_argument("--datasets_config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--experiments_dir", type=str, default="experiments_again")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default=None)

    # Execution control
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--filter-only", action="store_true",
                        help="Only submit filter/subset jobs")
    parser.add_argument("--train-only", action="store_true",
                        help="Only submit training jobs")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Cap total number of SLURM submissions")

    args = parser.parse_args()

    with open(args.baselines_config, 'r') as f:
        baselines_config = yaml.safe_load(f)
    with open(args.datasets_config, 'r') as f:
        datasets_config = yaml.safe_load(f)

    # Build sweep list from either experiment config or CLI args
    if args.experiment_config:
        with open(args.experiment_config, 'r') as f:
            experiment_config = yaml.safe_load(f)
        sweeps = experiment_config["experiments"]
    else:
        # CLI fallback: build a single sweep from args
        if not args.dataset_list or not args.baselines_list or not args.finetune_list:
            parser.error("Provide --experiment_config or all of "
                         "--dataset_list, --baselines_list, --finetune_list")
        sweeps = [{
            "name": "cli_sweep",
            "datasets": args.dataset_list,
            "baselines": args.baselines_list,
            "finetune_types": args.finetune_list,
            "lr_list": args.lr_list,
            "batch_sizes": args.batch_size_list,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
        }]

    # Generate and submit
    all_experiments = []
    for sweep in sweeps:
        sweep_name = sweep.get("name", "unnamed")
        experiments_dir = sweep.get("experiments_dir", args.experiments_dir)
        exps = list(generate_experiments(
            sweep, baselines_config, datasets_config,
            experiments_dir, args.datasets_config,
            args.wandb_project, args.wandb_entity, args.wandb_group,
            args.num_epochs, args.seed,
        ))
        print(f"Sweep '{sweep_name}': {len(exps)} experiments")
        all_experiments.extend(exps)

    print(f"Total experiments: {len(all_experiments)}\n")
    submit_jobs(all_experiments, dry_run=args.dry_run,
                filter_only=args.filter_only, train_only=args.train_only,
                max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
