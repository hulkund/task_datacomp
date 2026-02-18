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

source /data/vision/beery/scratch/neha/.bashrc
# conda activate datacomp

# python minibatxhing_ot.py
# python analyzing_runs_script.py 
# python OT_project/getting_diff_embeddings.py 
# python all_datasets/iWildCam/crop_megadetector_images.py
python OT_project/relabeling_algorithms/relabeling_iwildcam.py --dispatch
    
