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
│   └── subset_baselines.yaml  # Baseline method configs and parameters
├── DeepCore/              # GradMatch core selection
├── otdd/                  # Optimal transport dataset distance
├── trust/                 # Trust metrics and scoring
├── baselines.py           # Main CLI entry point for filters
├── make_subsets_with_config.py  # Full pipeline runner
└── requirements.txt
```

## Key Entry Points

1. **Apply a filter**: `python baselines.py --name <filter> --embedding_path <path> --save_path <output> --fraction <frac>`
2. **Train on subset**: `python training/train_on_subset.py --dataset_name <name> --subset_path <path> --dataset_config configs/datasets.yaml --finetune_type <type> --training_task <task> --outputs_path <path> --checkpoint_path <path>`
3. **Full pipeline**: `python make_subsets_with_config.py`
4. **SLURM batch**: `sbatch training/run_new_train.sh <dataset> <subset_path> <output> <config> <lr> <finetune_type> <batch_size> <checkpoint> <task>`

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
```

## Environment

- Runs on a SLURM cluster (MIT Vision group)
- Uses conda/pip with `requirements.txt`
- GPU required for training and embedding generation
