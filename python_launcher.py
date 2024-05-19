import os
import sys
import subprocess
import shlex
import torch
from os.path import exists
import glob
import yaml




        subprocess.call(shlex.split('sbatch run_cfg_eval_job.sh "%s"'%(cfg)))
