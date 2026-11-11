"""
EDA: NIH Chest X-ray14 — Long-tail Analysis
--------------------------------------------
Run this script locally (or as a quick cluster job) to explore the dataset
and produce all visualisations needed for your thesis/report.

Usage:
    python eda_chest_xray.py --data_dir /datasets/aharding/chest_xray14

All plots are saved to an 'eda_plots/' subfolder.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import Counter

# Import our dataset class to reuse its label-building logic
import sys
sys.path.append('/home-mscluster/aharding/chest_xray_longtail/src')
from datasets.chest_xray_dataset import PATHOLOGY_LABELS, NUM_CLASSES

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = sns.color_palette("muted")


def load_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load the master CSV and attach a binary column per pathology.
    This works entirely on the CSV — no images needed.
    """
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)

    # Build one binary column per pathology
    for label in PATHOLOGY_LABELS:
        # A label is present if it appears anywhere in the pipe-separated string
        df[label] = df["Finding Labels"].str.contains(label, regex=False).astype(int)

    # Useful derived columns
    df["num_labels"]   = df[PATHOLOGY_LABELS].sum(axis=1)  # labels per image
    df["no_finding"]   = (df["Finding Labels"] == "No Finding").astype(int)
    df["has_pathology"] = (df["num_labels"] > 0).astype(int)

    return df


def load_split_labels(data_dir: str):
    """Return sets of filenames for each official split."""
    def read_list(fname):
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())

    train_val = read_list("train_val_list.txt")
    test      = read_list("test_list.txt")
    return train_val, test


# ── Plot 1: Class frequency bar chart (sorted, linear scale) ─────────────────

def plot_class_frequency(df, out_dir):
    """
    The classic long-tail plot. Sort classes by frequency descending so
    the head-to-tail gradient is visually obvious.
    """
    counts = df[PATHOLOGY_LABELS].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=PALETTE[0], edgecolor="white", linewidth=0.5)

    # Annotate each bar with its count
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{int(val):,}",
                ha="center", va="bottom", fontsize=8)

    ax.set_title("Class Frequency — NIH Chest X-ray14\n"
                 "(sorted by frequency, showing long-tail distribution)")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Pathology")
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    path = os.path.join(out_dir, "01_class_frequency_linear.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: Log-scale version ─────────────────────────────────────────────────

def plot_class_frequency_log(df, out_dir):
    """
    Log scale makes the severity of the tail much clearer.
    Hernia (~227) vs Infiltration (~19,894) is a 87x difference —
    hard to see on linear scale but obvious on log.
    """
    counts = df[PATHOLOGY_LABELS].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values,
           color=PALETTE[1], edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("Class Frequency (log scale) — NIH Chest X-ray14\n"
                 "Log scale reveals the true severity of the long tail")
    ax.set_ylabel("Number of images (log scale)")
    ax.set_xlabel("Pathology")
    ax.tick_params(axis="x", rotation=40)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    path = os.path.join(out_dir, "02_class_frequency_log.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Imbalance ratio table ─────────────────────────────────────────────

def plot_imbalance_table(df, out_dir):
    """
    Shows the imbalance ratio for each class relative to the most common class.
    Also shows what fraction of images contain each pathology.
    This is the table you'll likely want in your thesis.
    """
    counts = df[PATHOLOGY_LABELS].sum().sort_values(ascending=False)
    total  = len(df)
    max_count = counts.max()

    summary = pd.DataFrame({
        "Pathology":        counts.index,
        "Count":            counts.values.astype(int),
        "% of dataset":     (counts.values / total * 100).round(2),
        "Imbalance ratio":  (max_count / counts.values).round(1),
    }).reset_index(drop=True)

    print("\n── Class Imbalance Summary ──────────────────────────────")
    print(summary.to_string(index=False))
    print(f"\nMost common: {counts.index[0]} ({int(counts.iloc[0]):,} images)")
    print(f"Rarest:      {counts.index[-1]} ({int(counts.iloc[-1]):,} images)")
    print(f"Imbalance ratio (max/min): {max_count / counts.iloc[-1]:.1f}x")

    # Save as CSV too — useful for tables in your report
    csv_path = os.path.join(out_dir, "imbalance_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Visual table as a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title("Class Imbalance Summary", pad=20, fontsize=13)

    plt.tight_layout()
    path = os.path.join(out_dir, "03_imbalance_table.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return summary


# ── Plot 4: Co-occurrence heatmap ─────────────────────────────────────────────

def plot_cooccurrence_heatmap(df, out_dir):
    """
    Shows how often pairs of pathologies appear together in the same image.

    WHY this matters for your project:
        NIH X-ray14 is MULTI-LABEL. Unlike single-label benchmarks, a patient
        can have Atelectasis AND Effusion simultaneously. The co-occurrence
        structure affects how well models can learn each class independently,
        and is particularly relevant for evaluating per-class AUC.
    """
    label_matrix = df[PATHOLOGY_LABELS].values  # (N, 14)

    # Co-occurrence matrix: entry (i,j) = number of images where both
    # pathology i and pathology j are present
    cooccur = label_matrix.T @ label_matrix  # (14, 14)

    # Normalise by the diagonal (i.e., by class frequency) to get
    # conditional probability P(j | i)
    freq = cooccur.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        cooccur_norm = cooccur / freq[:, None]
    np.fill_diagonal(cooccur_norm, 0)  # zero out self-co-occurrence

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cooccur_norm,
        xticklabels=PATHOLOGY_LABELS,
        yticklabels=PATHOLOGY_LABELS,
        annot=True, fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        cbar_kws={"label": "P(column | row)"},
    )
    ax.set_title("Pathology Co-occurrence — P(column pathology | row pathology)\n"
                 "Diagonal zeroed; values show conditional co-occurrence probability")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "04_cooccurrence_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Label cardinality distribution ────────────────────────────────────

def plot_label_cardinality(df, out_dir):
    """
    Histogram of how many labels each image has (0 = No Finding, 1, 2, ...).

    WHY this matters:
        Most multi-label benchmarks assume single-label — understanding how
        many labels co-occur per image is important for interpreting your
        results. A model that's good at single-pathology cases might still
        struggle with multi-pathology images.
    """
    # num_labels=0 means "No Finding" in our setup
    cardinality = df["num_labels"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cardinality.index.astype(str), cardinality.values,
           color=PALETTE[2], edgecolor="white")

    for x, y in zip(cardinality.index, cardinality.values):
        ax.text(str(x), y + 50, f"{y:,}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Label Cardinality Distribution\n"
                 "Number of pathology labels per image (0 = 'No Finding')")
    ax.set_xlabel("Number of labels per image")
    ax.set_ylabel("Number of images")

    plt.tight_layout()
    path = os.path.join(out_dir, "05_label_cardinality.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 6: Train / val / test split comparison ───────────────────────────────

def plot_split_distributions(df, train_val_files, test_files, out_dir):
    """
    Compares class frequencies across the official train+val and test splits.

    WHY this matters:
        You want to check the split is consistent — if tail classes are
        disproportionately absent from the test set, your AUC estimates
        will be unreliable for those classes.
    """
    df_trainval = df[df["Image Index"].isin(train_val_files)]
    df_test     = df[df["Image Index"].isin(test_files)]

    counts_tv   = df_trainval[PATHOLOGY_LABELS].sum()
    counts_test = df_test[PATHOLOGY_LABELS].sum()

    # Normalise to proportions within each split
    prop_tv   = counts_tv   / len(df_trainval)
    prop_test = counts_test / len(df_test)

    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, prop_tv.values,   width, label="Train+Val", color=PALETTE[0])
    ax.bar(x + width/2, prop_test.values, width, label="Test",      color=PALETTE[3])

    ax.set_xticks(x)
    ax.set_xticklabels(PATHOLOGY_LABELS, rotation=40, ha="right")
    ax.set_ylabel("Proportion of images with pathology")
    ax.set_title("Class Distribution: Train+Val vs Test Split\n"
                 "Consistent proportions indicate a reliable split")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    path = os.path.join(out_dir, "06_split_distributions.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 7: Patient-level statistics ─────────────────────────────────────────

def plot_patient_stats(df, out_dir):
    """
    Shows how many images per patient. Important because:
    - The same patient can have many X-rays over time.
    - Patient-level splitting prevents data leakage.
    - This plot motivates WHY we split by patient, not by image.
    """
    imgs_per_patient = df.groupby("Patient ID").size()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Distribution of images per patient
    axes[0].hist(imgs_per_patient.values, bins=30,
                 color=PALETTE[4], edgecolor="white")
    axes[0].set_title("Images per Patient")
    axes[0].set_xlabel("Number of images")
    axes[0].set_ylabel("Number of patients")

    # Cumulative: what % of patients have > N images?
    sorted_counts = np.sort(imgs_per_patient.values)[::-1]
    cumsum = np.cumsum(sorted_counts)
    axes[1].plot(cumsum / cumsum[-1] * 100, color=PALETTE[4])
    axes[1].set_title("Cumulative % of Images by Patient Rank\n"
                      "(shows skew: a few patients have many images)")
    axes[1].set_xlabel("Patient rank (sorted by image count)")
    axes[1].set_ylabel("Cumulative % of total images")
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter())

    print(f"\n── Patient Statistics ───────────────────────────────────")
    print(f"Total patients:          {len(imgs_per_patient):,}")
    print(f"Mean images per patient: {imgs_per_patient.mean():.1f}")
    print(f"Max images per patient:  {imgs_per_patient.max()}")
    print(f"Patients with >5 images: "
          f"{(imgs_per_patient > 5).sum():,} "
          f"({(imgs_per_patient > 5).mean()*100:.1f}%)")

    plt.tight_layout()
    path = os.path.join(out_dir, "07_patient_stats.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/datasets/aharding/chest_xray14",
                        help="Path to the NIH dataset root directory")
    parser.add_argument("--out_dir", type=str, default="eda_plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading metadata CSV...")
    df = load_metadata(args.data_dir)
    print(f"Total images: {len(df):,}")
    print(f"Unique patients: {df['Patient ID'].nunique():,}")

    print("\nLoading split file lists...")
    train_val_files, test_files = load_split_labels(args.data_dir)
    print(f"Train+val images: {len(train_val_files):,}")
    print(f"Test images:      {len(test_files):,}")

    print("\nGenerating plots...")
    plot_class_frequency(df, args.out_dir)
    plot_class_frequency_log(df, args.out_dir)
    plot_imbalance_table(df, args.out_dir)
    plot_cooccurrence_heatmap(df, args.out_dir)
    plot_label_cardinality(df, args.out_dir)
    plot_split_distributions(df, train_val_files, test_files, args.out_dir)
    plot_patient_stats(df, args.out_dir)

    print(f"\nAll plots saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
