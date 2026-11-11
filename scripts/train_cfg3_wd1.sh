#!/bin/bash
#SBATCH --job-name=cfg3_wd1
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/outputs/config3_wd1_stage1/slurm.out
#SBATCH --error=/home-mscluster/aharding/chest_xray_longtail/outputs/config3_wd1_stage1/slurm.err

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export WANDB_MODE=online

mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/config3_wd1_stage1
mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/config3_wd1_linear_probe
mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/config3_wd1_stage2

cd /home-mscluster/aharding/chest_xray_longtail

echo "========================================"
echo "CONFIG 3 WD=0.001 — STAGE 1"
echo "========================================"
python src/trainer.py --config configs/config3_wd1_stage1.yaml

if [ ! -f "outputs/config3_wd1_stage1/checkpoint_best.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found. Aborting."
    exit 1
fi

echo "========================================"
echo "CONFIG 3 WD=0.001 — LINEAR PROBE"
echo "========================================"
python src/linear_probe.py --config configs/config3_wd1_linear_probe.yaml

echo "========================================"
echo "CONFIG 3 WD=0.001 — STAGE 2"
echo "========================================"
python src/stage2_trainer.py --config configs/config3_wd1_stage2.yaml

echo "CONFIG 3 WD=0.001 COMPLETE"