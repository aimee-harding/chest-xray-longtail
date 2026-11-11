#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=mscluster44,mscluster59,mscluster62,mscluster74,mscluster75
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/logs/test_eval_%j.log

set -e

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.9

cd /home-mscluster/aharding/chest_xray_longtail

# GPU check: abort instead of silently running on CPU
echo "Node: $(hostname)"
python -c "import torch, sys; ok = torch.cuda.is_available(); print('CUDA available:', ok); sys.exit(0 if ok else 1)" \
    || { echo "ERROR: No GPU visible to PyTorch. Aborting."; exit 1; }

python src/evaluate_test.py --n-boot 1000

echo "TEST EVALUATION COMPLETE"
