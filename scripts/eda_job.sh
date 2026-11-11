#!/bin/bash
#SBATCH --job-name=eda_xray
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/outputs/eda.out
#SBATCH --error=/home-mscluster/aharding/chest_xray_longtail/outputs/eda.err

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

pip install seaborn --quiet

python ~/chest_xray_longtail/notebooks/eda_chest_xray.py \
    --data_dir /datasets/aharding/chest_xray14 \
    --out_dir /home-mscluster/aharding/chest_xray_longtail/outputs/eda_plots
