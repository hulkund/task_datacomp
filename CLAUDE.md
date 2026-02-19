<<<<<<< HEAD
# Task DataComp

A benchmarking framework for evaluating **data subset selection methods** on diverse downstream tasks. The core idea: given a large training pool and a target task, select the best subset of training data and evaluate how well a model trained on that subset performs.

## High-Level Pipeline

The system runs a 3-stage pipeline, orchestrated by `make_subsets_with_config.py`:

```
1. FILTER/SELECT  -->  2. TRAIN  -->  3. EVALUATE
  (baselines.py)      (training/)     (training/)
```

Each stage is submitted as a SLURM job on the `vision-beery` partition. The conda environment is `unlabeled_exp`.

### Stage 1: Data Subset Selection (`baselines.py` -> `baselines/apply_filter_ours.py`)

Selects a subset of training UIDs from a data pool using a filtering method. Output: `.npy` file of selected UIDs.

**Entry point:** `python baselines.py --name <method> --embedding_path <path> --save_path <path> [--fraction F] [--threshold T]`
**SLURM wrapper:** `run_baseline.sh` (embedding-based methods), `run_csv_baseline.sh` (label-based methods)

### Stage 2: Training (`training/train_on_subset.py`)

Trains a model on the selected subset.

**Entry point:** `python training/train_on_subset.py --dataset_name <name> --subset_path <path> --training_task <task> --finetune_type <type> ...`
**SLURM wrapper:** `training/run_new_train.sh`

### Stage 3: Evaluation (also `training/train_on_subset.py`)

After training, the same script loads the best checkpoint and evaluates on each test set, saving per-task metrics as JSON.

## Orchestration

**`make_subsets_with_config.py`** is the main experiment runner. It iterates over the Cartesian product of:
- Datasets (e.g., `SelfDrivingCar`, `iWildCam`, `GeoDE`, ...)
- Baselines/filter methods
- Finetune types (`linear_probe`, `full_finetune_resnet50`, `full_finetune_resnet101`, `lora_finetune_vit`)
- Learning rates, batch sizes
- Fractions (what portion of the pool to keep)
- Tasks (each dataset has multiple test tasks, e.g., `test1`-`test4`)

For each combination, it submits SLURM jobs for filtering and training (skipping if output already exists).

## Directory Structure

```
task_datacomp/
├── baselines.py                  # CLI entry point for subset selection
├── make_subsets_with_config.py   # Experiment orchestrator (submits SLURM jobs)
├── metrics_df.py                 # Aggregates experiment JSON results into tables
├── gradmatch.py                  # GradMatch coreset selection (standalone)
│
├── configs/
│   ├── baselines.yaml            # Per-method config (fraction_list, task scope)
│   └── datasets.yaml             # Per-dataset config (paths, splits, task type)
│
├── baselines/                    # Subset selection methods
│   ├── apply_filter_ours.py      # Main filter dispatcher (routes --name to filter)
│   ├── basic_csv_baselines.py    # Label-matching baselines (match_label, match_dist)
│   ├── get_embeddings.py         # Compute CLIP/DINO embeddings for a dataset
│   ├── model_backbone.py         # Model factory (ResNet50/101, CLIP ViT, LoRA ViT)
│   ├── utils.py                  # Dataset loader, embedding utils, FAISS index, metrics
│   └── filters/                  # Individual filter implementations
│       ├── basic_filter.py       #   Language/length/caption heuristics
│       ├── caption_filter.py     #   Caption-based filtering
│       ├── centroids_filter.py   #   Image/text centroid matching (image_based, text_based, image_clip)
│       ├── clip_filter.py        #   CLIP similarity score threshold/fraction
│       ├── gradmatch_filter.py   #   Gradient matching coreset selection
│       ├── image_align_filter.py #   Image embedding alignment to val set
│       ├── random_filter.py      #   Random subset selection
│       ├── text_alignment_filter.py # Text embedding alignment to val set
│       ├── tsds_filter.py        #   Task-specific data selection
│       └── utils.py              #   Shared: load_embedding, load_uids
│
├── training/                     # Model training and evaluation
│   ├── train_on_subset.py        # Main training entry point (dispatches by task type)
│   ├── train_engine.py           # TrainEngine class (routes train/evaluate to task-specific code)
│   ├── train_on_subset_classification.py  # Classification training (linear probe, full finetune)
│   ├── train_on_subset_regression.py      # Regression training
│   ├── train_on_subset_reid.py            # Re-identification training
│   ├── train_on_subset_fixmatch.py        # Semi-supervised FixMatch (pseudolabeling)
│   ├── train_on_query.py                  # Train on query/val set directly
│   └── run_new_train.sh                   # SLURM wrapper for training
│
├── all_datasets/                 # Dataset implementations (one file per dataset)
│   ├── task_dataset.py           # Base TaskDataset class
│   ├── iWildCam_dataset.py       # Camera trap wildlife classification
│   ├── GeoDE_dataset.py          # Geographic diversity evaluation
│   ├── AutoArborist_dataset.py   # Urban tree species classification
│   ├── SelfDrivingCar_dataset.py # Autonomous driving (regression)
│   ├── FMoW_dataset.py           # Functional Map of the World
│   ├── CropHarvest_dataset.py    # Crop type mapping
│   ├── FishDetection_dataset.py  # Underwater fish detection
│   ├── ReID_dataset.py           # Animal re-identification
│   └── COOS_dataset.py           # Coral reef classification
│
├── tune_baselines/               # Hyperparameter tuning for filter methods
├── visualizations/               # Visualization utilities
├── DeepCore/                     # External: coreset selection library (datasets, methods, nets)
├── experiments_again/            # Experiment output directory (metrics JSONs, subsets)
├── slurm/                        # SLURM job logs
│
├── run_baseline.sh               # SLURM: run embedding-based filter
├── run_csv_baseline.sh           # SLURM: run label-based filter
├── run_getting_embeddings_and_centroids.sh  # SLURM: precompute embeddings
├── getting_oracle_nums.sh        # Utility: count oracle set sizes
└── getting_query_nums.sh         # Utility: count query set sizes
```

## Available Filter Methods

| Method | Key | Needs Val Embeddings | Description |
|--------|-----|---------------------|-------------|
| No filter | `no_filter` | No | Use all training data |
| Random | `random_filter` | No | Random fraction of pool |
| CLIP score | `clip_score` | No | Keep top-k% by CLIP similarity |
| Image-based | `image_based` | Yes | Centroid matching on image embeddings |
| Text-based | `text_based` | Yes | Centroid matching on text embeddings |
| Image+CLIP | `image_clip` | Yes | Centroid matching + CLIP score |
| Image alignment | `image_alignment` | Yes | Cosine similarity to val image embeddings |
| Text alignment | `text_alignment` | Yes | Cosine similarity to val text embeddings |
| TSDS | `tsds` | Yes | Task-specific data selection |
| GradMatch | `gradmatch` | Yes | Gradient matching coreset selection |
| Match label | `match_label` | Yes (labels) | Keep training samples whose labels appear in val |
| Match dist | `match_dist` | Yes (labels) | Resample training to match val label distribution |

## Datasets

Configured in `configs/datasets.yaml`. Each dataset has:
- `csv_root_path`: Path to CSV split files
- `img_root_path`: Path to images
- `task_list`: List of test tasks (typically `test1`-`test4`)
- `training_task`: One of `classification`, `regression`, `detection`, `reid`

Supported: GeoDE, AutoArborist, iWildCam, SelfDrivingCar, FMoW, FishDetection, ReID, CropHarvest, COOS

## Data Flow

1. **Embeddings** (precomputed `.npy` files): Each training/val sample has `image_embedding`, `text_embedding`, `similarity_score`, `uid`, `label`
2. **Filter output** (`.npy` file of UIDs): Selected subset indices
3. **Training**: Loads dataset, filters by UIDs, trains model with early stopping
4. **Metrics** (`.json`): Per-task accuracy, class-average accuracy, or MSE

## Key Classes

- **`TaskDataset`** (`all_datasets/task_dataset.py`): Base dataset class. Loads CSV splits, filters by subset UIDs.
- **`TrainEngine`** (`training/train_engine.py`): Dispatches `train()` and `evaluate()` to task-specific implementations.
- **`FaissIndexIVFFlat`** (`baselines/utils.py`): FAISS approximate nearest neighbor index for embedding search.

## Environment

- **Cluster:** MIT CSAIL vision-beery SLURM partition
- **Conda env:** `unlabeled_exp`
- **GPU:** Most jobs request 1 GPU, 8 CPUs, 100GB RAM
- **Data root:** `/data/vision/beery/scratch/neha/task-datacomp/all_datasets/`
=======
# CLAUDE.md

## Project Overview

**task-datacomp** is a data subset selection and filtering benchmark for computer vision. It evaluates different data filtering/selection strategies and their impact on model performance when training on reduced datasets across diverse tasks (classification, regression, detection, re-identification).

## Repository Structure

```
task-datacomp/
├── all_datasets/          # Dataset classes and data (iWildCam, GeoDE, AutoArborist, SelfDrivingCar, FishDetection, ReID, CropHarvest)
├── baselines/             # Baseline selection methods
│   ├── filters/           # Modular filter implementations (clip, gradmatch, tsds, random, etc.)
│   ├── apply_filter_ours.py  # Filter application router
│   └── utils.py           # Dataset loading, metrics
├── training/              # Training scripts
│   ├── train_on_subset.py              # Main dispatcher (routes by task type)
│   ├── train_on_subset_classification.py
│   ├── train_on_subset_regression.py
│   ├── train_on_subset_reid.py
│   ├── train_on_subset_fixmatch.py
│   ├── train_engine.py    # TrainEngine class
│   ├── model_backbone.py  # Model loading
│   └── run_new_train.sh   # SLURM submission script
├── configs/
│   ├── datasets.yaml      # Dataset paths and task definitions
│   ├── subset_baselines.yaml  # Baseline method configs and parameters
│   └── experiments.yaml   # Experiment sweep definitions for batch runs
├── DeepCore/              # GradMatch core selection
├── otdd/                  # Optimal transport dataset distance
├── trust/                 # Trust metrics and scoring
├── baselines.py           # Main CLI entry point for filters
├── make_subsets_with_config.py       # Full pipeline runner (single sweep)
├── make_config_experiments.py        # Batch experiment runner (multiple sweeps, dry-run support)
└── requirements.txt
```

## Key Entry Points

1. **Apply a filter**: `python baselines.py --name <filter> --embedding_path <path> --save_path <output> --fraction <frac>`
2. **Train on subset**: `python training/train_on_subset.py --dataset_name <name> --subset_path <path> --dataset_config configs/datasets.yaml --finetune_type <type> --training_task <task> --outputs_path <path> --checkpoint_path <path>`
3. **Full pipeline (single sweep)**: `python make_subsets_with_config.py`
4. **Batch experiments (multiple sweeps)**: `python make_config_experiments.py --experiment_config configs/experiments.yaml`
5. **SLURM batch**: `sbatch training/run_new_train.sh <dataset> <subset_path> <output> <config> <lr> <finetune_type> <batch_size> <checkpoint> <task>`

## Filter Names

`no_filter`, `random_filter`, `image_based`, `text_based`, `image_clip`, `clip_score`, `image_alignment`, `text_alignment`, `tsds`, `gradmatch`, `match_label`, `match_dist`

## Datasets

`iWildCam`, `GeoDE`, `CropHarvest`, `AutoArborist`, `SelfDrivingCar`, `ReID`, `FishDetection`

**Finetune types**: `linear_probe`, `lora_finetune_vit`, `full_finetune_resnet50`, `full_finetune_resnet101`

**Training tasks**: `classification`, `regression`, `detection`, `reid`

## Tech Stack

- **Python 3.x** with PyTorch, torchvision, transformers, timm
- **CLIP** (OpenAI) for embeddings and scoring
- **FAISS** for vector similarity search
- **PEFT/LoRA** for efficient fine-tuning
- **ultralytics** (YOLOv5/v9) for detection tasks
- **wandb** for experiment tracking
- **pandas/numpy/scikit-learn** for data processing
- **YAML** for configuration

## Conventions

- Dataset classes follow `<Name>Dataset` naming (e.g., `iWildCamDataset`) and inherit from `TaskDataset`
- Filter functions follow `load_uids_with_<method_name>` naming
- DataFrames use columns: `uid`, `label`, `filename`
- Embeddings stored as `.npy` files; subsets stored as `.npy` uid arrays
- `sys.path.append('/data/vision/beery/scratch/neha/task-datacomp/')` is used for imports across modules
- Device selection: `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`
- Configs loaded via `yaml.safe_load()`

## Common Commands

```bash
# Generate embeddings and centroids
./run_getting_embeddings_and_centroids.sh

# Run a baseline filter
python baselines.py --name clip_score --embedding_path <path> --save_path <output> --fraction 0.5

# Train on a filtered subset
python training/train_on_subset.py --dataset_name iWildCam --subset_path <path> \
    --dataset_config configs/datasets.yaml --finetune_type lora_finetune_vit \
    --training_task classification --outputs_path <path> --checkpoint_path <path>

# Batch experiments from a sweep config (preview without submitting)
python make_config_experiments.py --experiment_config configs/experiments.yaml --dry-run

# Submit only filter jobs, then only training jobs (two-stage workflow)
python make_config_experiments.py --experiment_config configs/experiments.yaml --filter-only
python make_config_experiments.py --experiment_config configs/experiments.yaml --train-only

# Quick CLI one-off (no config file needed)
python make_config_experiments.py \
    --dataset_list iWildCam GeoDE --baselines_list clip_score no_filter \
    --finetune_list lora_finetune_vit --dry-run

# Cap total SLURM submissions
python make_config_experiments.py --experiment_config configs/experiments.yaml --max-jobs 50
```

## Environment

- Runs on a SLURM cluster (MIT Vision group)
- Uses conda/pip with `requirements.txt`
- GPU required for training and embedding generation
>>>>>>> master
