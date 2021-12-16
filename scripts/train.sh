#!/usr/bin/env bash
pyenv activate conv_onet 

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

python "$GIT_ROOT"/convolutional_occupancy_networks/train.py "$CONFIG" --weights "$WEIGHTS"

echo "Job ended at $(date)"
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Took " ${total_time} " s"
