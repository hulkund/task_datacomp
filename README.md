# task-datacomp

A data subset selection and filtering benchmark for computer vision. Evaluates different data filtering/selection strategies and their impact on model performance when training on reduced datasets across diverse tasks (classification, regression, detection, re-identification).

## Setup

Requires a SLURM cluster with GPU access (MIT Vision group).

```bash
pip install -r requirements.txt
```

## Running Experiments

### 1. Generate embeddings and centroids

Before running any filters that use embeddings, generate them for your datasets:

```bash
./run_getting_embeddings_and_centroids.sh
```

### 2. Run experiments with a config file (recommended)

Define your experiment sweeps in a YAML config file (see `configs/experiments.yaml` for an example):

```yaml
experiments:
  - name: "classification_sweep"
    datasets: [iWildCam, GeoDE, AutoArborist]
    baselines: [no_filter, clip_score, image_alignment]
    finetune_types: [lora_finetune_vit, full_finetune_resnet50]
    lr_list: [0.001, 0.0001]
    batch_sizes: [128]
    num_epochs: 100

  - name: "quick_test"
    datasets: [iWildCam]
    baselines: [no_filter]
    finetune_types: [lora_finetune_vit]
    num_epochs: 2
```

Preview what will be submitted without launching any jobs:

```bash
python make_config_experiments.py --experiment_config configs/experiments.yaml --dry-run
```

Submit all jobs:

```bash
python make_config_experiments.py --experiment_config configs/experiments.yaml
```

#### Useful flags

| Flag | Description |
|------|-------------|
| `--dry-run` | Print commands without submitting |
| `--filter-only` | Only submit subset selection jobs |
| `--train-only` | Only submit training jobs |
| `--max-jobs N` | Cap total number of SLURM submissions |
| `--num_epochs N` | Override default epoch count (default: 100) |

#### Two-stage workflow

Filter jobs typically finish faster than training. You can run them in two stages:

```bash
# Stage 1: submit all filter jobs
python make_config_experiments.py --experiment_config configs/experiments.yaml --filter-only

# Stage 2: once filters finish, submit training jobs
python make_config_experiments.py --experiment_config configs/experiments.yaml --train-only
```

Both stages automatically skip jobs whose outputs already exist.

### 3. Quick CLI one-off (no config file)

For ad-hoc runs without writing a config file:

```bash
python make_config_experiments.py \
    --dataset_list iWildCam GeoDE \
    --baselines_list clip_score no_filter \
    --finetune_list lora_finetune_vit \
    --num_epochs 50 \
    --dry-run
```

### 4. Run a single filter manually

```bash
python baselines.py --name clip_score \
    --embedding_path all_datasets/iWildCam/embeddings/train_embeddings.npy \
    --save_path experiments/iWildCam/clip_score_0.5/subset.npy \
    --fraction 0.5
```

### 5. Train on a single subset manually

```bash
python training/train_on_subset.py \
    --dataset_name iWildCam \
    --subset_path experiments/iWildCam/clip_score_0.5/subset.npy \
    --dataset_config configs/datasets.yaml \
    --finetune_type lora_finetune_vit \
    --training_task classification \
    --outputs_path experiments/iWildCam/clip_score_0.5/ \
    --checkpoint_path experiments/iWildCam/clip_score_0.5/checkpoint \
    --num_epochs 100
```

Or via SLURM:

```bash
sbatch training/run_new_train.sh iWildCam \
    experiments/iWildCam/clip_score_0.5/subset.npy \
    experiments/iWildCam/clip_score_0.5/ \
    configs/datasets.yaml 0.001 lora_finetune_vit 128 \
    experiments/iWildCam/clip_score_0.5/checkpoint \
    classification "" "" "" "" 100
```

## Experiment config reference

Each sweep in `experiments.yaml` supports:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Label for the sweep (used in log output) |
| `datasets` | Yes | List of datasets to run |
| `baselines` | Yes | List of filter methods to apply |
| `finetune_types` | Yes | List of finetuning strategies |
| `lr_list` | No | Learning rates (default: `[0.001]`) |
| `batch_sizes` | No | Batch sizes (default: `[128]`) |
| `num_epochs` | No | Training epochs (default: `100`) |
| `experiments_dir` | No | Output directory (default: `experiments_again`) |

The script generates the cartesian product of all list fields. Fractions and per-task splits come from `configs/subset_baselines.yaml` and `configs/datasets.yaml`.

## Available datasets

| Dataset | Training task |
|---------|--------------|
| `iWildCam` | classification |
| `GeoDE` | classification |
| `AutoArborist` | classification |
| `SelfDrivingCar` | regression |
| `FishDetection` | detection |
| `ReID` | re-identification |

## Available filters

`no_filter`, `random_filter`, `image_based`, `text_based`, `image_clip`, `clip_score`, `image_alignment`, `text_alignment`, `tsds`, `gradmatch`, `match_label`, `match_dist`

## Available finetune types

`linear_probe`, `lora_finetune_vit`, `full_finetune_resnet50`, `full_finetune_resnet101`
