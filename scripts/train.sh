#!/usr/bin/env bash
SLURM_HOME="/net/rmc-gpu03/home_local/humt_ma"
export PYENV_ROOT="$SLURM_HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="$PYENV_ROOT/shims:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
if command -v pyenv >/dev/null; then eval "$(pyenv init -)"; fi
export PYENV_VIRTUALENV_DISABLE_PROMPT=1

pyenv activate conv_onet 

USERSTORE="/volume/USERSTORE/humt_ma"
export LD_LIBRARY_PATH="$USERSTORE/glibc/build/math:$LD_LIBRARY_PATH"

echo "=====Job Infos ===="
echo "Node List: " "$SLURM_NODELIST"
echo "Job ID: " "$SLURM_JOB_ID"
echo "Job Name:" "$SLURM_JOB_NAME"
echo "Partition: " "$SLURM_JOB_PARTITION"
echo "Submit directory:" "$SLURM_SUBMIT_DIR"
echo "Submit host:" "$SLURM_SUBMIT_HOST"
echo "In the directory: $(pwd)"
echo "As the user: $(whoami)"
echo "Python version: $(python -c 'import sys; print(sys.version)')"
echo "pip version: $(pip --version)"

nvidia-smi

start_time=$(date +%s)
echo "Job Started at $(date)"

CONFIG="$1"
WEIGHTS="$2"

echo "config:" "$CONFIG"
echo "weights:" "${WEIGHTS:=''}"

cd /net/rmc-lx0038/home_local/git/convolutional_occupancy_networks
python train.py "$CONFIG" --weights "$WEIGHTS"

echo "Job ended at $(date)"
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Took " ${total_time} " s"
