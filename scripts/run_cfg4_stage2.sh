#!/bin/bash
#SBATCH --job-name=cfg4_s2
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster59
#SBATCH --time=24:00:00
#SBATCH --output=logs/cfg4_s2_%j.log

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

cd /home-mscluster/aharding/chest_xray_longtail

if [ ! -f "outputs/config4_ral_stage1/checkpoint_best.pt" ]; then
    echo "ERROR: Stage 1 checkpoint not found. Aborting."
    exit 1
fi

echo "=== Linear Probe ==="
python src/linear_probe.py --config configs/config4_ral_linear_probe.yaml

echo "=== Stage 2: Classifier retraining ==="
python src/stage2_trainer.py --config configs/config4_ral_stage2.yaml

echo "=== Done ==="