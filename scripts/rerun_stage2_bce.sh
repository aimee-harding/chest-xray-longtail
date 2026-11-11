#!/bin/bash
#SBATCH --job-name=s2_bce_fix
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/outputs/stage2_bce/slurm_rerun.out
#SBATCH --error=/home-mscluster/aharding/chest_xray_longtail/outputs/stage2_bce/slurm_rerun.err

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export WANDB_MODE=online

cd /home-mscluster/aharding/chest_xray_longtail

# Remove old metrics so they don't append to previous run
rm -f outputs/stage2_bce/metrics.csv
rm -f outputs/stage2_bce/metrics.json

python src/stage2_trainer.py --config configs/stage2_bce.yaml
