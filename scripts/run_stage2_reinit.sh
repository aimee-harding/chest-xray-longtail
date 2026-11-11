#!/bin/bash
# One Stage 2 (cRT) run. Usage:  sbatch scripts/run_stage2_reinit.sh configs/stage2_bce_reinit.yaml
#SBATCH --job-name=s2_reinit
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=mscluster44,mscluster59,mscluster62,mscluster74,mscluster75
#SBATCH --time=08:00:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/logs/s2_reinit_%j.log
set -e
source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9
export WANDB_MODE=online
cd /home-mscluster/aharding/chest_xray_longtail
echo "Node: $(hostname)   Config: $1"
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU'); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: No GPU visible to PyTorch. Aborting."; exit 1; }
python -u src/stage2_trainer.py --config "$1"
echo "DONE $1"
