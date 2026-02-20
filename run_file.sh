#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/analyzing_jobs-%J.out
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=1-23:00:00
#SBATCH --requeue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.local_paths.sh" ]; then
    source "$SCRIPT_DIR/.local_paths.sh"
fi

BASHRC_PATH="${BASHRC_PATH:-$HOME/.bashrc}"
if [ -f "$BASHRC_PATH" ]; then
    source "$BASHRC_PATH"
fi
# conda activate datacomp

# python minibatxhing_ot.py
# python analyzing_runs_script.py 
# python OT_project/getting_diff_embeddings.py 
# python all_datasets/iWildCam/crop_megadetector_images.py
python OT_project/relabeling_algorithms/relabeling_iwildcam.py --dispatch
    
