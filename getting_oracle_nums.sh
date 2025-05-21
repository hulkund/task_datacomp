#!/bin/bash
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm-%J.out
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

source /data/vision/beery/scratch/neha/.bashrc
conda activate datacomp

learning_rates=(0.1 0.01 0.001)   # Fixed missing space between values
batch_sizes=(32)     
dataset_names=(AutoArborist)
# Loop through hyperparameters and submit separate jobs
for dataset_name in "${dataset_names[@]}"
do 
    for lr in "${learning_rates[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do
            for filename in experiments/$dataset_name/oracle_subsets/*.npy; do
            echo "$filename"
            # Create a job script for each combination of hyperparameters
                job_file="job_file/job_lr_${lr}_bs_${batch_size}.slurm"
                
                echo "#!/bin/bash" > $job_file
                echo "#SBATCH --partition=vision-beery" >> $job_file
                echo "#SBATCH --qos=vision-beery-main" >> $job_file
                echo "#SBATCH --account=vision-beery" >> $job_file
                echo "#SBATCH --gres=gpu:1" >> $job_file
                echo "#SBATCH --output=slurm/oracle-%J.out" >> $job_file
                echo "#SBATCH -c 8" >> $job_file
                echo "#SBATCH --mem=100G" >> $job_file
                echo "#SBATCH --time=12:00:00" >> $job_file
                echo "source /data/vision/beery/scratch/neha/.bashrc" >> $job_file
                echo "conda activate datacomp" >> $job_file
                echo "python baselines/train_on_subset_classification.py --outputs_path experiments/$dataset_name/oracle_subsets/ --batch_size $batch_size --lr $lr --subset_path \"$filename\" --dataset_name $dataset_name --finetune_type full_finetune --dataset_config configs/datasets.yaml  --checkpoint_path oracle_checkpoint" >> $job_file
                
                # Submit the job
                sbatch $job_file
            done
        done
        
    done
done
