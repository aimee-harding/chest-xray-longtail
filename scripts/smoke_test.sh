#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/home-mscluster/aharding/chest_xray_longtail/outputs/smoke_test.out
#SBATCH --error=/home-mscluster/aharding/chest_xray_longtail/outputs/smoke_test.err

source /home-mscluster/aharding/miniconda3/etc/profile.d/conda.sh
conda activate chestxray

python - <<'EOF'
import sys
sys.path.append('/home-mscluster/aharding/chest_xray_longtail/src')

from datasets.chest_xray_dataset import ChestXray14Dataset, PATHOLOGY_LABELS
from torch.utils.data import DataLoader

DATA_DIR = '/datasets/aharding/chest_xray14'

print("=" * 50)
print("Loading train split...")
train_ds = ChestXray14Dataset(DATA_DIR, split='train')

print("\nLoading val split...")
val_ds = ChestXray14Dataset(DATA_DIR, split='val')

print("\nLoading test split...")
test_ds = ChestXray14Dataset(DATA_DIR, split='test')

print("\n" + "=" * 50)
print(f"Train size: {len(train_ds):,}")
print(f"Val size:   {len(val_ds):,}")
print(f"Test size:  {len(test_ds):,}")
print(f"Total:      {len(train_ds)+len(val_ds)+len(test_ds):,}")

print("\n" + "=" * 50)
print("Testing DataLoader — loading one batch...")
loader = DataLoader(train_ds, batch_size=16, num_workers=4, shuffle=True)
images, labels = next(iter(loader))
print(f"Image batch shape: {images.shape}")   # expect (16, 3, 224, 224)
print(f"Label batch shape: {labels.shape}")   # expect (16, 14)
print(f"Image min/max:     {images.min():.3f} / {images.max():.3f}")
print(f"Labels sample:\n{labels[:4]}")

print("\n" + "=" * 50)
print("Class frequencies (train):")
freqs = train_ds.get_class_frequencies()
for label, freq in sorted(zip(PATHOLOGY_LABELS, freqs), key=lambda x: -x[1]):
    print(f"  {label:<22} {int(freq):>6,}")

print("\npos_weight (first 5):", train_ds.get_pos_weight()[:5])
print("\nSmoke test PASSED!")
EOF
