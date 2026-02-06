from pathlib import Path

slurm_dir = Path("/data/vision/beery/scratch/evelyn/task_datacomp/slurm")
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_fraction_0.5_lam_0.5_random_seed_0_selection_batch_16_selection_lr_0.01"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_fraction_0.5_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/glister_eta_0.1_fraction_0.1_random_seed_0_selection_batch_16_selection_lr_0.01/test4_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/glister_eta_0.1_fraction_0.05_random_seed_1_selection_batch_16_selection_lr_0.01/test4_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/AutoArborist/glister_eta_0.1_fraction_0.9_random_seed_42_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.25_lam_0.5_random_seed_42_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.25_lam_0.5_random_seed_42_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "RuntimeError: CUDA error: uncorrectable ECC error encountered"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.25_lam_0.5_random_seed_0_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.25_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.9_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.75_lam_0.5_random_seed_42_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_fraction_0.5_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/GeoDE/gradmatch_acf_fraction_0.75_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test1_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/GeoDE/gradmatch_acf_fraction_0.75_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test1_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/AutoArborist/gradmatch_acf_fraction_0.25_lam_0.5_random_seed_0_selection_batch_16_selection_lr_0.01/test1_subset.npy"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/AutoArborist/gradmatch_fraction_0.25_lam_0.5_random_seed_0_selection_batch_16_selection_lr_0.01/test2_subset.npy"
# pattern = "TIME"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/gradmatch_acf_fraction_0.25_lam_0.5_random_seed_42_selection_batch_16_selection_lr_0.01/test4_time.txt"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/iWildCam/random_filter_fraction_0.05_random_seed_0/test2_finetune=full_finetune_resnet50_lr=0.001_batchsize=128"
# pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/GeoDE/gradmatch_acf_fraction_0.9_lam_0.5_random_seed_1_selection_batch_16_selection_lr_0.01/test1_subset.npy"
pattern = "/data/vision/beery/scratch/evelyn/task_datacomp/experiments/GeoDE/gradmatch_acf_fraction_0.9_lam_0.5_random_seed_0_selection_batch_16_selection_lr_0.01/test1_subset.npy"


matches = []
for logfile in slurm_dir.glob("slurm-*.out"):
    with logfile.open("r", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            if pattern in line:
                print(f"[MATCH] {logfile}")
                matches.append(logfile)
                break

print("="*20)
print("Number of matches:", len(matches))


for match in sorted(matches):
    print(match)