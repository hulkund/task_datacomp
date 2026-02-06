# main_scripts

Pipeline scripts for running subset selection and training experiments via SLURM.

## Prerequisites

Before running any script, make sure you have:

- Pre-computed embeddings in `all_datasets/{dataset}/embeddings/` (`train_embeddings.npy`, `val1_embeddings.npy`, `test1_embeddings.npy`, etc.)
   - To get embeddings for baselines: Run the file ./run_getting_clip_embeddings.sh

- Dataset config at `configs/datasets.yaml` (for iWildCam, Auto Arborist, GeoDE)
- Sweep parameters for baseline methods at `main_scripts/config.yaml`

## Pipeline overview

```
1. train_warmstart.py       (only needed for gradmatch, gradmatch_acf, glister)
        |
        v
2. create_subsets.py        (run all baseline selection methods)
        |
        v
3. train_subsets.py         (fine-tune models on selected subsets)
        |
        v
4. time_methods.py          (optional: measure selection runtime)
```

## How to run

All scripts are run from the repo root:

```bash
python main_scripts/<script>.py
```

Each script has a `# --- Configuration ---` section at the top. Before running, edit:

- `BASELINES` — which selection methods to run
- `DATASET_LIST` — uncomment the dataset/split tuples you want
- Sweep parameters (fractions, seeds, etc.) in `config.yaml`

## Output structure

```
experiments/{dataset}/{method}_{params}/
  {split}_subset.npy                                       # selected subset indices
  {split}_time.txt                                         # timing benchmark (optional)
  {split}_{finetune}_lr={lr}_batchsize={bs}_metrics.json   # evaluation metrics
```

Warmstart checkpoints are stored separately:

```
experiments/{dataset}/{val_split}/{method}_{model}_epochs={n}_seed={s}/
  warmstart_weights.pth
```

## Baseline methods

| Method | Notes |
|--------|-------|
| `no_filter` | Use full dataset (fraction=1) |
| `random_filter` | Random sampling |
| `clip_score` | CLIP embedding scoring |
| `match_dist` | Distance matching (CSV-based, uses `run_csv_baseline.sh`) |
| `tsds` | Temporal-Spatial Distance Sampling |
| `gradmatch` | Gradient matching (can use warmstart) |
| `gradmatch_acf` | Gradient matching with adaptive class fraction (can use warmstart) |
| `glister` | GLISTER coreset selection (can use warmstart) |
