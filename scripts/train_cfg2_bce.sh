#!/bin/bash
#SBATCH --job-name=cfg2_bce
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/outputs/cfg2_bce/slurm.out
#SBATCH --error=/home-mscluster/aharding/chest_xray_longtail/outputs/cfg2_bce/slurm.err

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export WANDB_MODE=online

mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/cfg2_bce
mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/stage1_bce
mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/linear_probe_bce
mkdir -p /home-mscluster/aharding/chest_xray_longtail/outputs/stage2_bce

cd /home-mscluster/aharding/chest_xray_longtail

# ── Stage 1: Train backbone end-to-end with plain BCE ─────────────────────────
echo "============================================"
echo "STAGE 1: BCE backbone training"
echo "============================================"
python src/trainer.py --config configs/stage1_bce.yaml

# Check Stage 1 completed successfully
if [ ! -f "outputs/stage1_bce/checkpoint_best.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found. Aborting."
    exit 1
fi
echo "Stage 1 complete."

# ── Linear Probe: Evaluate Stage 1 representations ────────────────────────────
echo "============================================"
echo "LINEAR PROBE: Evaluating Stage 1 representations"
echo "============================================"
python src/linear_probe.py --config configs/linear_probe_bce.yaml
echo "Linear probe complete."

# ── Stage 2: Classifier retraining with class-aware sampling ──────────────────
echo "============================================"
echo "STAGE 2: Classifier retraining"
echo "============================================"
python src/stage2_trainer.py --config configs/stage2_bce.yaml
echo "Stage 2 complete."

echo "============================================"
echo "CONFIG 2 PIPELINE COMPLETE"
echo "============================================"