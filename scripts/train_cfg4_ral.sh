#!/bin/bash
#SBATCH --job-name=cfg4_ral
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/logs/cfg4_ral_gpu_%j.log

# Stop the job as soon as any step fails (so a crashed step can't be missed)
set -e

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export WANDB_MODE=online

cd /home-mscluster/aharding/chest_xray_longtail

# ── GPU check: abort immediately instead of silently training on CPU ─────────
echo "Node: $(hostname)"
nvidia-smi || true
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU'); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: No GPU visible to PyTorch. Aborting."; exit 1; }

mkdir -p outputs/config4_ral_stage1 outputs/config4_ral_linear_probe outputs/config4_ral_stage2

echo "========================================"
echo "CONFIG 4 RAL — STAGE 1"
echo "========================================"
python src/trainer.py --config configs/config4_ral_stage1.yaml

if [ ! -f "outputs/config4_ral_stage1/checkpoint_best.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found. Aborting."
    exit 1
fi

echo "========================================"
echo "CONFIG 4 RAL — LINEAR PROBE"
echo "========================================"
python src/linear_probe.py --config configs/config4_ral_linear_probe.yaml

echo "========================================"
echo "CONFIG 4 RAL — STAGE 2"
echo "========================================"
python src/stage2_trainer.py --config configs/config4_ral_stage2.yaml

echo "CONFIG 4 RAL COMPLETE"