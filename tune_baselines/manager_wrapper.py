import os
import subprocess
import time
import yaml
from pathlib import Path

USER_NAME = "evelynz"

def get_specific_job_count(MAX_QUEUE_SIZE, TARGET_JOB_NAME):
    """Counts only jobs matching the specific name."""
    try:
        # -n filters by job name, -u by user, -h removes header
        cmd = ["squeue", "-u", USER_NAME, "-n", TARGET_JOB_NAME, "-h", "-t", "PD,R"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = [line for line in result.stdout.split('\n') if line.strip()]
        return len(lines)
    except Exception as e:
        print(f"Error checking queue: {e}")
        return MAX_QUEUE_SIZE


def run_manager(MAX_QUEUE_SIZE, POLL_INTERVAL, TARGET_JOB_NAME, jobs_to_submit, run=False):
    print(f"Collected {len(jobs_to_submit)} jobs. Starting managed submission...")

    submitted = 0
    total = len(jobs_to_submit)

    while jobs_to_submit:
        current_active = get_specific_job_count(MAX_QUEUE_SIZE, TARGET_JOB_NAME)
        
        if current_active < MAX_QUEUE_SIZE:
            num_to_spawn = MAX_QUEUE_SIZE - current_active
            for _ in range(num_to_spawn):
                if not jobs_to_submit:
                    break
                
                cmd = jobs_to_submit.pop(0)
                command = ["sbatch", "--job-name", TARGET_JOB_NAME] + cmd
                
                if run:
                    subprocess.run(command)
                
                submitted += 1
                print(f"[{submitted}/{total}] Submitted. '{TARGET_JOB_NAME}' active: {current_active + 1}")
                current_active += 1
        else:
            print(f"Queue full for '{TARGET_JOB_NAME}' ({current_active}/{MAX_QUEUE_SIZE}). Waiting...")
            time.sleep(POLL_INTERVAL)

    print("All baseline jobs submitted.")