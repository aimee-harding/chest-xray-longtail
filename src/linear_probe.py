"""
Linear Probing Evaluation
--------------------------
Evaluates the quality of Stage 1 representations BEFORE Stage 2
classifier retraining. This is a key part of your proposal:

    "Representation quality is additionally assessed via linear probing
     before classifier retraining, enabling the contribution of Stage 1
     to be separated from Stage 2 effects."

What linear probing does:
    1. Load the Stage 1 backbone checkpoint
    2. FREEZE the entire backbone (no updates at all)
    3. Train a fresh linear classifier on top of the frozen features
    4. Evaluate on the val set

WHY linear probing?
    The AUC after Stage 2 reflects BOTH the quality of the Stage 1
    representations AND the effect of Stage 2 retraining. Linear probing
    isolates the Stage 1 contribution — if one configuration's linear
    probe AUC is higher, its backbone learned better representations,
    independent of any Stage 2 effects.

    A good linear probe score means the features are already well-organised
    in the 1024-dim embedding space, even before classifier retraining.

Usage:
    python linear_probe.py --config configs/linear_probe_bce.yaml
"""

import os
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.append('/home-mscluster/aharding/chest_xray_longtail/src')

from datasets.chest_xray_dataset import (
    ChestXray14Dataset,
    PATHOLOGY_LABELS,
    NUM_CLASSES,
)
from models.densenet import get_backbone
from trainer import load_config, get_scheduler, evaluate


class LinearProbe:
    """
    Trains and evaluates a linear classifier on top of frozen Stage 1 features.

    The backbone is completely frozen — not even the original classification
    head is used. A fresh Linear(1024, 14) is trained from scratch on top
    of the fixed feature extractor.

    This gives a clean measure of how well the Stage 1 representations
    support the downstream task, independent of the head initialisation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LinearProbe] Using device: {self.device}")

        # ── Datasets ──────────────────────────────────────────────────────
        data_cfg = config["data"]
        self.train_ds = ChestXray14Dataset(
            data_cfg["data_dir"], split="train",
            val_frac=data_cfg["val_frac"], seed=data_cfg["seed"],
        )
        self.val_ds = ChestXray14Dataset(
            data_cfg["data_dir"], split="val",
            val_frac=data_cfg["val_frac"], seed=data_cfg["seed"],
        )

        batch_size  = config["training"]["batch_size"]
        num_workers = data_cfg["num_workers"]

        # Instance-balanced sampling for linear probe training
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
        )

        # ── Load backbone and freeze it completely ─────────────────────────
        self.backbone = get_backbone(num_classes=14, pretrained=False).to(self.device)

        checkpoint_path = config["experiment"]["stage1_checkpoint"]
        print(f"[LinearProbe] Loading Stage 1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from epoch {checkpoint['epoch']} "
              f"(val_auc={checkpoint['val_auc']:.4f})")

        # Freeze EVERYTHING in the backbone including the original head
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("[LinearProbe] Backbone fully frozen")

        # ── Fresh linear classifier ────────────────────────────────────────
        # A new Linear(1024, 14) trained from scratch on frozen features.
        # This is the "probe" — it tests whether the features are linearly
        # separable for each pathology class.
        self.probe = nn.Linear(1024, NUM_CLASSES).to(self.device)
        nn.init.xavier_uniform_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

        # ── Loss and optimiser ─────────────────────────────────────────────
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=config["training"]["optimizer"]["lr"],
            weight_decay=config["training"]["optimizer"].get("weight_decay", 0.0),
        )
        self.scheduler = get_scheduler(self.optimizer, config)

        # ── Output ────────────────────────────────────────────────────────
        self.save_dir = config["output"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        wandb.init(
            project="chest-xray-longtail",
            name=config["experiment"]["name"],
            config={
                "stage":             "linear_probe",
                "stage1_checkpoint": checkpoint_path,
                "lr":                config["training"]["optimizer"]["lr"],
                "epochs":            config["training"]["epochs"],
            }
        )

        self.best_val_auc      = 0.0
        self.epochs_no_improve = 0
        self.early_stop_patience = config["training"]["early_stopping"]["patience"]
        self.metrics_log       = []

    # ── Extract features ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pass images through the frozen backbone to get 1024-dim embeddings.
        Uses get_features() which returns embeddings before the classifier head.
        """
        return self.backbone.get_features(images)

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.probe.train()
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Extract frozen features — no gradients for the backbone
            features = self._extract_features(images)   # (B, 1024)

            # Train only the probe
            self.optimizer.zero_grad()
            logits = self.probe(features)               # (B, 14)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch} [{i+1}/{n_batches}] "
                      f"loss={loss.item():.4f}")

        return total_loss / n_batches

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> dict:
        self.probe.eval()
        all_labels = []
        all_probs  = []
        total_loss = 0.0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            features = self._extract_features(images)
            logits   = self.probe(features)
            loss     = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs  = np.concatenate(all_probs,  axis=0)

        metrics = evaluate(all_labels, all_probs, threshold=0.5)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        epochs = self.config["training"]["epochs"]
        print(f"\n[LinearProbe] Training linear probe for {epochs} epochs")

        for epoch in range(1, epochs + 1):
            train_loss  = self._train_epoch(epoch)
            val_results = self._validate()
            val_auc     = val_results["auc_macro"]

            print(f"\nEpoch {epoch}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_macro_auc={val_auc:.4f} | "
                  f"val_macro_f1={val_results['f1_macro']:.4f}")

            TAIL = {"Pneumonia", "Hernia", "Fibrosis"}
            print(f"\n  {'Class':<22} {'AUC':>6} {'F1':>6} "
                  f"{'Sens':>6} {'Spec':>6}")
            print(f"  {'-'*52}")
            for i, label in enumerate(PATHOLOGY_LABELS):
                marker = " *" if label in TAIL else ""
                print(f"  {label:<22} "
                      f"{val_results['auc_per_class'][i]:>6.4f} "
                      f"{val_results['f1_per_class'][i]:>6.4f} "
                      f"{val_results['sensitivity_per_class'][i]:>6.4f} "
                      f"{val_results['specificity_per_class'][i]:>6.4f}{marker}")

            # Log to W&B
            wandb.log({
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_results["loss"],
                "auc_macro":  val_auc,
                "f1_macro":   val_results["f1_macro"],
                **{f"auc_{label}": val_results["auc_per_class"][i]
                   for i, label in enumerate(PATHOLOGY_LABELS)},
            })

            # Save metrics
            self.metrics_log.append({
                "epoch":      epoch,
                "train_loss": train_loss,
                "val": {
                    "auc_macro": val_auc,
                    "f1_macro":  val_results["f1_macro"],
                    "per_class": {
                        label: {
                            "auc":         val_results["auc_per_class"][i],
                            "f1":          val_results["f1_per_class"][i],
                            "sensitivity": val_results["sensitivity_per_class"][i],
                            "specificity": val_results["specificity_per_class"][i],
                        }
                        for i, label in enumerate(PATHOLOGY_LABELS)
                    },
                },
            })
            with open(os.path.join(self.save_dir, "linear_probe_metrics.json"), "w") as f:
                json.dump(self.metrics_log, f, indent=2)

            self.scheduler.step(val_auc)

            if val_auc > self.best_val_auc:
                self.best_val_auc      = val_auc
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stop_patience:
                    print(f"\n[LinearProbe] Early stopping at epoch {epoch}")
                    break

        # Save final probe results summary
        summary = {
            "best_val_auc_macro": self.best_val_auc,
            "stage1_checkpoint":  self.config["experiment"]["stage1_checkpoint"],
            "per_class_auc": {
                label: val_results["auc_per_class"][i]
                for i, label in enumerate(PATHOLOGY_LABELS)
            },
        }
        with open(os.path.join(self.save_dir, "linear_probe_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[LinearProbe] Done. Best val macro AUC: {self.best_val_auc:.4f}")
        wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[Main] Linear probing: {config['experiment']['name']}")

    probe = LinearProbe(config)
    probe.run()