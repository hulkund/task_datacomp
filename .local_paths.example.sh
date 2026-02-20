#!/bin/bash
# Copy this file to .local_paths.sh and set your local values.
# .local_paths.sh is git-ignored.

# Shell init file used by job scripts before activating envs.
export BASHRC_PATH="$HOME/.bashrc"

# Conda environment name or absolute env path.
export CONDA_ENV="datacomp"

# Micromamba environment name for scripts that use micromamba.
export MAMBA_ENV="datacomp"
