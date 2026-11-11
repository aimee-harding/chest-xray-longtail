#!/bin/bash
#SBATCH --job-name=joint_bce
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/outputs/joint_bce_baseline/slurm.out
#SBATCH --error=/home-mscluster/aharding/chest_xray_longtail/outputs/joint_bce_baseline/slurm.err

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

# CUDA setup
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9

# W&B — try online, fall back to offline if cluster blocks outbound connections
export WANDB_MODE=online

# Create output directory
mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/joint_bce_baseline

cd /home-mscluster/aharding/chest_xray_longtail

python src/trainer.py \
    --config configs/joint_bce_baseline.yaml
