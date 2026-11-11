# Long-tailed multi-label chest X-ray classification

Code for my BSc Honours research report (University of the Witwatersrand, 2026):
a comparison of Stage 1 representation learning strategies within the decoupled
training framework on NIH ChestX-ray14.

## Configurations
- C1: joint BCE · C2: decoupled BCE · C3: BCE with weight decay sweep
- C4: Robust Asymmetric Loss · C5: simplified multi-label supervised contrastive loss (MSC)
- Stage 2: classifier re-training (cRT) with class-aware sampling; warm-start ablation

## Layout
- `src/` – dataset, models, losses (`losses/ral_loss.py`, `losses/msc_loss.py`), trainers, linear probe, test evaluation
- `configs/` – one YAML file per stage and configuration
- `scripts/` – SLURM job scripts and helpers
- `tests/` – check of the MSC loss against a direct implementation of the paper's equations
- `outputs/test_eval/` – final test-set results (summary, per-class, paired comparisons, thresholds)

## Data
NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC

Set `data.data_dir` in the configs to the dataset folder (containing `Data_Entry_2017.csv`,
`train_val_list.txt`, `test_list.txt` and the `images_*` folders).

## Running

```
pip install -r requirements.txt
python src/trainer.py --config configs/stage1_bce.yaml                      # Stage 1
python src/linear_probe.py --config configs/linear_probe_bce.yaml           # linear probe
python src/stage2_trainer.py --config configs/stage2_bce_reinit.yaml        # Stage 2 (cRT)
python src/contrastive_trainer.py --config configs/config5_msc_stage1.yaml  # C5 Stage 1
python src/evaluate_test.py                                                 # test evaluation
```
