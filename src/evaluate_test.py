"""
Final test-set evaluation for all configurations.

For every checkpoint:
  1. Run the validation set once and pick a per-class threshold that maximises
     F1 on validation (the test set is never used for any choice).
  2. Run the test set once and report:
       - per-class and macro AUC, with 95% bootstrap confidence intervals
       - sensitivity / specificity / F1 at 0.5 and at the tuned thresholds
  3. Paired bootstrap comparisons between key configurations (the same
     resampled test sets are used for every checkpoint, so differences
     between two models can be tested directly).

Predicted probabilities are cached, so re-running the analysis (e.g. with a
different number of bootstrap samples) does not need the GPU again.

Usage (from ~/chest_xray_longtail):
    python src/evaluate_test.py
    python src/evaluate_test.py --n-boot 2000          # more bootstrap samples
    python src/evaluate_test.py --recompute            # ignore cached predictions
    python src/evaluate_test.py --only c2_s2,c3_wd2e4_s2
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.chest_xray_dataset import ChestXray14Dataset, PATHOLOGY_LABELS, NUM_CLASSES  # noqa: E402
from models.densenet import get_backbone  # noqa: E402
from trainer import evaluate  # noqa: E402  (your existing metric function)


ROOT = "/home-mscluster/aharding/chest_xray_longtail"
OUT = os.path.join(ROOT, "outputs")

# Same data settings as every training config, so the val split is identical.
DATA_DIR = "/datasets/aharding/chest_xray14"
VAL_FRAC = 0.1
SEED = 42

TAIL_CLASSES = ["Hernia", "Pneumonia", "Fibrosis"]

# name -> (description, checkpoint path relative to the outputs folder)
CHECKPOINTS = {
    "c1_joint":     ("Config 1 joint BCE",          "joint_bce_baseline/checkpoint_best.pt"),
    "c2_s1":        ("Config 2 BCE WD 1e-4 S1",     "stage1_bce/checkpoint_best.pt"),
    "c2_s2":        ("Config 2 BCE WD 1e-4 S2",     "stage2_bce/checkpoint_best.pt"),
    "c3_wd2e4_s1":  ("Config 3 WD 2e-4 S1",         "config3_wd4_stage1/checkpoint_best.pt"),
    "c3_wd2e4_s2":  ("Config 3 WD 2e-4 S2",         "config3_wd4_stage2/checkpoint_best.pt"),
    "c3_wd5e4_s1":  ("Config 3 WD 5e-4 S1",         "config3_wd5_stage1/checkpoint_best.pt"),
    "c3_wd5e4_s2":  ("Config 3 WD 5e-4 S2",         "config3_wd5_stage2/checkpoint_best.pt"),
    "c3_wd1e3_s1":  ("Config 3 WD 1e-3 S1",         "config3_wd1_stage1/checkpoint_best.pt"),
    "c3_wd1e3_s2":  ("Config 3 WD 1e-3 S2",         "config3_wd1_stage2/checkpoint_best.pt"),
    "c3_wd5e3_s1":  ("Config 3 WD 5e-3 S1",         "config3_wd2_stage1/checkpoint_best.pt"),
    "c3_wd5e3_s2":  ("Config 3 WD 5e-3 S2",         "config3_wd2_stage2/checkpoint_best.pt"),
    "c3_wd1e2_s1":  ("Config 3 WD 1e-2 S1",         "config3_wd3_stage1/checkpoint_best.pt"),
    "c3_wd1e2_s2":  ("Config 3 WD 1e-2 S2",         "config3_wd3_stage2/checkpoint_best.pt"),
    "c4_ral_s1":    ("Config 4 RAL S1",             "config4_ral_stage1/checkpoint_best.pt"),
    "c4_ral_s2":    ("Config 4 RAL S2",             "config4_ral_stage2/checkpoint_best.pt"),
    "c5_msc_s1":    ("Config 5 MSC S1",             "config5_msc_stage1/checkpoint_best.pt"),
    "c5_msc_s2":    ("Config 5 MSC S2",             "config5_msc_stage2/checkpoint_best.pt"),
    "c2_s2r": ("Config 2 BCE S2 cRT", "stage2_bce_reinit/checkpoint_best.pt"),
    "c3_wd2e4_s2r": ("Config 3 WD 2e-4 S2 cRT", "config3_wd4_stage2_reinit/checkpoint_best.pt"),
    "c3_wd5e4_s2r": ("Config 3 WD 5e-4 S2 cRT", "config3_wd5_stage2_reinit/checkpoint_best.pt"),
    "c3_wd1e3_s2r": ("Config 3 WD 1e-3 S2 cRT", "config3_wd1_stage2_reinit/checkpoint_best.pt"),
    "c3_wd5e3_s2r": ("Config 3 WD 5e-3 S2 cRT", "config3_wd2_stage2_reinit/checkpoint_best.pt"),
    "c3_wd1e2_s2r": ("Config 3 WD 1e-2 S2 cRT", "config3_wd3_stage2_reinit/checkpoint_best.pt"),
    "c4_ral_s2r": ("Config 4 RAL S2 cRT", "config4_ral_stage2_reinit/checkpoint_best.pt"),
    "c5_msc_s2r": ("Config 5 MSC S2 cRT", "config5_msc_stage2_reinit/checkpoint_best.pt"),
}

# Paired comparisons: (A, B) reports AUC(A) - AUC(B) with a 95% CI.
COMPARISONS = [
    ("c2_s1", "c1_joint"),          # does decoupled Stage 1 differ from joint training?
    ("c2_s2", "c1_joint"),          # full decoupling vs joint baseline
    ("c2_s2", "c2_s1"),             # effect of Stage 2 on a BCE backbone
    ("c3_wd2e4_s1", "c2_s1"),       # tuned weight decay, Stage 1
    ("c3_wd2e4_s2", "c2_s2"),       # tuned weight decay, Stage 2
    ("c3_wd2e4_s2", "c3_wd2e4_s1"), # effect of Stage 2 on the WD 2e-4 backbone
    ("c4_ral_s1", "c2_s1"),         # RAL vs BCE, Stage 1
    ("c4_ral_s2", "c2_s2"),         # RAL vs BCE, Stage 2
    ("c4_ral_s2", "c4_ral_s1"),     # effect of Stage 2 on the RAL backbone
    ("c5_msc_s1", "c2_s1"),         # contrastive vs BCE representation, Stage 1
    ("c5_msc_s2", "c2_s2"),         # contrastive vs BCE, after Stage 2
    ("c5_msc_s2", "c5_msc_s1"),     # effect of Stage 2 on the contrastive backbone
    ("c5_msc_s1", "c4_ral_s1"),     # contrastive vs RAL, Stage 1
    ("c2_s2r", "c2_s1"),
    ("c3_wd2e4_s2r", "c3_wd2e4_s1"),
    ("c4_ral_s2r", "c4_ral_s1"),
    ("c5_msc_s2r", "c5_msc_s1"),
    ("c2_s2r", "c2_s2"),
    ("c3_wd2e4_s2r", "c3_wd2e4_s2"),
    ("c4_ral_s2r", "c4_ral_s2"),
    ("c5_msc_s2r", "c5_msc_s2"),
    ("c3_wd2e4_s2r", "c2_s2r"),
    ("c4_ral_s2r", "c2_s2r"),
    ("c5_msc_s2r", "c2_s2r"),
    ("c2_s2r", "c1_joint"),
]


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        logits = model(images.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def load_model(path, device):
    model = get_backbone(num_classes=NUM_CLASSES, pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("epoch"), ckpt.get("val_auc")


def get_predictions(name, path, loaders, device, cache_dir, recompute):
    """Returns {'val': (probs, labels), 'test': (probs, labels)}, using the cache if present."""
    cache = os.path.join(cache_dir, f"{name}.npz")
    if os.path.exists(cache) and not recompute:
        d = np.load(cache)
        print(f"  using cached predictions: {cache}")
        return {"val": (d["val_probs"], d["val_labels"]), "test": (d["test_probs"], d["test_labels"])}, None

    model, epoch, val_auc = load_model(path, device)
    print(f"  loaded checkpoint (epoch {epoch}, stored val_auc={val_auc})")
    out = {split: predict(model, loader, device) for split, loader in loaders.items()}
    np.savez_compressed(cache,
                        val_probs=out["val"][0], val_labels=out["val"][1],
                        test_probs=out["test"][0], test_labels=out["test"][1])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, (epoch, val_auc)


# ── Thresholds (chosen on validation only) ───────────────────────────────────

def f1_optimal_thresholds(labels, probs):
    thr = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        if labels[:, c].sum() == 0:
            continue
        precision, recall, t = precision_recall_curve(labels[:, c], probs[:, c])
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best = int(np.nanargmax(f1[:-1]))  # last point has no threshold
        thr[c] = t[best]
    return thr


# ── Bootstrap ────────────────────────────────────────────────────────────────

def auc_matrix(labels, probs):
    """Per-class AUC via the rank formula (identical to roc_auc_score, handles ties).
    labels/probs: (N, C). Returns (C,), NaN where a class has no positives or negatives."""
    ranks = rankdata(probs, axis=0)
    n_pos = labels.sum(axis=0)
    n_neg = labels.shape[0] - n_pos
    rank_sum = (ranks * labels).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def bootstrap_aucs(labels, probs, boot_idx):
    """Returns (B, C) array of per-class AUCs over bootstrap resamples."""
    out = np.empty((len(boot_idx), labels.shape[1]))
    for b, idx in enumerate(boot_idx):
        out[b] = auc_matrix(labels[idx], probs[idx])
    return out


def ci(samples):
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--recompute", action="store_true", help="ignore cached predictions")
    ap.add_argument("--only", type=str, default="", help="comma-separated checkpoint names")
    ap.add_argument("--outputs", type=str, default=OUT, help="folder containing the training outputs")
    ap.add_argument("--data-dir", type=str, default=DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=None, help="default: <outputs>/test_eval")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    args.out_dir = args.out_dir or os.path.join(args.outputs, "test_eval")
    ckpts = {k: (d, os.path.join(args.outputs, rel)) for k, (d, rel) in CHECKPOINTS.items()}
    cache_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_test] device: {device}")
    if device.type != "cuda":
        print("[evaluate_test] WARNING: no GPU — inference will be slow (cached checkpoints are fine).")

    names = [n for n in ckpts if not args.only or n in args.only.split(",")]
    missing = [n for n in names if not os.path.exists(ckpts[n][1])
               and not os.path.exists(os.path.join(cache_dir, f"{n}.npz"))]
    for n in missing:
        print(f"[evaluate_test] SKIPPING {n}: checkpoint not found at {ckpts[n][1]}")
    names = [n for n in names if n not in missing]

    loaders = None
    def get_loaders():
        ds = {
            "val":  ChestXray14Dataset(args.data_dir, split="val",  val_frac=VAL_FRAC, seed=SEED),
            "test": ChestXray14Dataset(args.data_dir, split="test", val_frac=VAL_FRAC, seed=SEED),
        }
        return {k: DataLoader(v, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True) for k, v in ds.items()}

    tail_idx = [PATHOLOGY_LABELS.index(c) for c in TAIL_CLASSES]
    boot_idx = None
    summary_rows, class_rows, thresholds, boot_macro, boot_tail = [], [], {}, {}, {}

    for name in names:
        desc, path = ckpts[name]
        print(f"\n=== {name}: {desc} ===")
        needs_inference = args.recompute or not os.path.exists(os.path.join(cache_dir, f"{name}.npz"))
        if needs_inference and loaders is None:
            loaders = get_loaders()
        preds, _ = get_predictions(name, path, loaders, device, cache_dir, args.recompute)
        val_p, val_y = preds["val"]
        test_p, test_y = preds["test"]

        if boot_idx is None:
            rng = np.random.default_rng(args.boot_seed)
            n = len(test_y)
            boot_idx = rng.integers(0, n, size=(args.n_boot, n))  # shared by every checkpoint
        elif len(test_y) != boot_idx.shape[1]:
            raise RuntimeError("Test set size differs between checkpoints — check the data paths.")

        # thresholds from validation only
        thr = f1_optimal_thresholds(val_y, val_p)
        thresholds[name] = dict(zip(PATHOLOGY_LABELS, map(float, thr)))

        val_auc = evaluate(val_y, val_p)["auc_macro"]
        m05 = evaluate(test_y, test_p, threshold=0.5)
        mt = evaluate(test_y, test_p, threshold=thr)   # per-class thresholds broadcast across columns

        point = auc_matrix(test_y, test_p)
        boots = bootstrap_aucs(test_y, test_p, boot_idx)
        macro_b = np.nanmean(boots, axis=1)
        tail_b = np.nanmean(boots[:, tail_idx], axis=1)
        boot_macro[name], boot_tail[name] = macro_b, tail_b

        macro_lo, macro_hi = ci(macro_b)
        tail_lo, tail_hi = ci(tail_b)
        summary_rows.append({
            "checkpoint": name, "description": desc,
            "val_auc_macro": round(val_auc, 4),
            "test_auc_macro": round(float(np.nanmean(point)), 4),
            "test_auc_macro_ci_lo": round(macro_lo, 4), "test_auc_macro_ci_hi": round(macro_hi, 4),
            "test_auc_tail": round(float(np.nanmean(point[tail_idx])), 4),
            "test_auc_tail_ci_lo": round(tail_lo, 4), "test_auc_tail_ci_hi": round(tail_hi, 4),
            "sens_macro@0.5": round(m05["sensitivity_macro"], 4),
            "spec_macro@0.5": round(m05["specificity_macro"], 4),
            "f1_macro@0.5": round(m05["f1_macro"], 4),
            "sens_macro@tuned": round(mt["sensitivity_macro"], 4),
            "spec_macro@tuned": round(mt["specificity_macro"], 4),
            "f1_macro@tuned": round(mt["f1_macro"], 4),
        })
        for c, cls in enumerate(PATHOLOGY_LABELS):
            lo, hi = ci(boots[:, c])
            class_rows.append({
                "checkpoint": name, "class": cls, "tail": cls in TAIL_CLASSES,
                "n_pos_test": int(test_y[:, c].sum()),
                "auc": round(float(point[c]), 4), "auc_ci_lo": round(lo, 4), "auc_ci_hi": round(hi, 4),
                "threshold": round(float(thr[c]), 4),
                "sens@0.5": round(m05["sensitivity_per_class"][c], 4),
                "spec@0.5": round(m05["specificity_per_class"][c], 4),
                "f1@0.5": round(m05["f1_per_class"][c], 4),
                "sens@tuned": round(mt["sensitivity_per_class"][c], 4),
                "spec@tuned": round(mt["specificity_per_class"][c], 4),
                "f1@tuned": round(mt["f1_per_class"][c], 4),
            })
        print(f"  test macro AUC {np.nanmean(point):.4f} [{macro_lo:.4f}, {macro_hi:.4f}]  "
              f"tail AUC {np.nanmean(point[tail_idx]):.4f} [{tail_lo:.4f}, {tail_hi:.4f}]")
        print(f"  macro sens/spec/F1  @0.5: {m05['sensitivity_macro']:.3f}/{m05['specificity_macro']:.3f}/{m05['f1_macro']:.3f}"
              f"   @tuned: {mt['sensitivity_macro']:.3f}/{mt['specificity_macro']:.3f}/{mt['f1_macro']:.3f}")

    # paired comparisons
    comp_rows = []
    for a, b in COMPARISONS:
        if a not in boot_macro or b not in boot_macro:
            continue
        for metric, store in (("macro", boot_macro), ("tail", boot_tail)):
            diff = store[a] - store[b]
            lo, hi = ci(diff)
            pa = next(r for r in summary_rows if r["checkpoint"] == a)[f"test_auc_{metric}"]
            pb = next(r for r in summary_rows if r["checkpoint"] == b)[f"test_auc_{metric}"]
            comp_rows.append({
                "A": a, "B": b, "metric": f"{metric}_auc",
                "A_auc": pa, "B_auc": pb, "diff_A_minus_B": round(pa - pb, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "significant_95": bool(lo > 0 or hi < 0),
                "p_boot_two_sided": round(float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))), 4),
            })

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.DataFrame(class_rows).to_csv(os.path.join(args.out_dir, "per_class.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(os.path.join(args.out_dir, "comparisons.csv"), index=False)
    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n[evaluate_test] wrote summary.csv, per_class.csv, comparisons.csv, thresholds.json → {args.out_dir}")
    if comp_rows:
        print("\nPaired comparisons (A − B, 95% bootstrap CI):")
        for r in comp_rows:
            flag = "*" if r["significant_95"] else " "
            print(f"  {flag} {r['A']:>12} vs {r['B']:<12} {r['metric']:<9} {r['diff_A_minus_B']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
