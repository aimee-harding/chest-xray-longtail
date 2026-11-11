
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import csv
import json

import sys
sys.path.append('/home-mscluster/aharding/chest_xray_longtail/src')

from datasets.chest_xray_dataset import (
    ChestXray14Dataset,
    PATHOLOGY_LABELS,
    NUM_CLASSES,
)
from models.densenet import get_backbone, freeze_backbone
from trainer import (
    load_config,
    get_loss_fn,
    get_optimizer,
    get_scheduler,
    evaluate,
)


class Stage2Trainer:
    """
    Classifier retraining stage of the decoupled framework.

    Key differences from Stage 1 Trainer:
      - Backbone is frozen (only head is trained)
      - Class-aware sampling replaces instance-balanced sampling
      - Higher learning rate (only optimising a single linear layer)
      - Loads weights from a Stage 1 checkpoint
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Stage2Trainer] Using device: {self.device}")

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

        # ── DataLoaders ───────────────────────────────────────────────────
        batch_size  = config["training"]["batch_size"]
        num_workers = data_cfg["num_workers"]

        # Class-aware sampling — key Stage 2 mechanism.
        # Builds a WeightedRandomSampler that oversamples tail classes.
        # Each image is weighted by 1/frequency_of_its_rarest_label.
        # This means images containing Hernia or Pneumonia appear far
        # more often per epoch than their raw frequency would suggest.
        print("[Stage2Trainer] Building class-aware sampler...")
        sampler = self.train_ds.get_class_aware_sampler()

        # shuffle=False because the sampler handles ordering
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # ── Model ─────────────────────────────────────────────────────────
        # Load architecture (pretrained=False since we load from checkpoint)
        model_cfg = config["model"]
        self.model = get_backbone(
            num_classes=model_cfg["num_classes"],
            pretrained=False,
        ).to(self.device)

        # Load Stage 1 weights
        checkpoint_path = config["experiment"]["stage1_checkpoint"]
        print(f"[Stage2Trainer] Loading Stage 1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # cRT (Kang et al., 2020): discard the Stage 1 classifier and learn a new one from scratch.
        # Same initialisation as a new DenseNet121Backbone head.
        if config["model"].get("reinit_classifier", False):
            torch.nn.init.xavier_uniform_(self.model.classifier.weight)
            torch.nn.init.zeros_(self.model.classifier.bias)
            print("[Stage2Trainer] Classifier re-initialised (cRT)")
        else:
            print("[Stage2Trainer] Classifier initialised from Stage 1")
        print(f"  Loaded from epoch {checkpoint['epoch']} "
              f"(val_auc={checkpoint['val_auc']:.4f})")

        # Freeze backbone — only the linear head will be trained
        freeze_backbone(self.model)

        # ── Loss ──────────────────────────────────────────────────────────
        # Plain BCE — class-aware sampling handles imbalance instead of
        # pos_weight. Using both simultaneously would double-correct.
        pos_weight = self.train_ds.get_pos_weight().to(self.device)
        self.criterion = get_loss_fn(config, pos_weight)

        # ── Optimiser and scheduler ───────────────────────────────────────
        # get_optimizer filters to requires_grad=True params only,
        # so only the linear head gets updated (backbone is frozen).
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        # ── Output directory ──────────────────────────────────────────────
        self.save_dir = config["output"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # ── W&B ───────────────────────────────────────────────────────────
        wandb.init(
            project="chest-xray-longtail",
            name=config["experiment"]["name"],
            config={
                "batch_size":        config["training"]["batch_size"],
                "lr":                config["training"]["optimizer"]["lr"],
                "weight_decay":      config["training"]["optimizer"]["weight_decay"],
                "loss":              config["training"]["loss"]["type"],
                "epochs":            config["training"]["epochs"],
                "sampling":          config["training"]["sampling"],
                "stage":             config["experiment"]["stage"],
                "stage1_checkpoint": config["experiment"]["stage1_checkpoint"],
            }
        )

        # ── Training state ────────────────────────────────────────────────
        self.best_val_auc      = 0.0
        self.epochs_no_improve = 0
        self.early_stop_patience = config["training"]["early_stopping"]["patience"]
        self.metrics_log       = []
        self.metrics_path      = os.path.join(self.save_dir, "metrics.csv")
        self.metrics_json_path = os.path.join(self.save_dir, "metrics.json")

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
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
        self.model.eval()
        all_labels = []
        all_probs  = []
        total_loss = 0.0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs  = np.concatenate(all_probs,  axis=0)

        metrics = evaluate(all_labels, all_probs, threshold=0.5)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    # ── Metrics saving ────────────────────────────────────────────────────────

    def _save_metrics(self, epoch: int, train_loss: float,
                      val_results: dict) -> None:
        row = {
            "epoch":             epoch,
            "train_loss":        round(train_loss, 6),
            "val_loss":          round(val_results["loss"], 6),
            "auc_macro":         round(val_results["auc_macro"], 6),
            "f1_macro":          round(val_results["f1_macro"], 6),
            "sensitivity_macro": round(val_results["sensitivity_macro"], 6),
            "specificity_macro": round(val_results["specificity_macro"], 6),
            "accuracy":          round(val_results["accuracy"], 6),
        }
        for i, label in enumerate(PATHOLOGY_LABELS):
            row[f"auc_{label}"]         = round(val_results["auc_per_class"][i], 6)
            row[f"f1_{label}"]          = round(val_results["f1_per_class"][i], 6)
            row[f"sensitivity_{label}"] = round(val_results["sensitivity_per_class"][i], 6)
            row[f"specificity_{label}"] = round(val_results["specificity_per_class"][i], 6)

        write_header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self.metrics_log.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val": {
                "loss":              val_results["loss"],
                "auc_macro":         val_results["auc_macro"],
                "f1_macro":          val_results["f1_macro"],
                "sensitivity_macro": val_results["sensitivity_macro"],
                "specificity_macro": val_results["specificity_macro"],
                "accuracy":          val_results["accuracy"],
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
        with open(self.metrics_json_path, "w") as f:
            json.dump(self.metrics_log, f, indent=2)

        wandb.log({
            "epoch":             epoch,
            "train_loss":        train_loss,
            "val_loss":          val_results["loss"],
            "auc_macro":         val_results["auc_macro"],
            "f1_macro":          val_results["f1_macro"],
            "sensitivity_macro": val_results["sensitivity_macro"],
            "specificity_macro": val_results["specificity_macro"],
            "accuracy":          val_results["accuracy"],
            **{f"auc_{label}": val_results["auc_per_class"][i]
               for i, label in enumerate(PATHOLOGY_LABELS)},
            **{f"f1_{label}": val_results["f1_per_class"][i]
               for i, label in enumerate(PATHOLOGY_LABELS)},
            **{f"sensitivity_{label}": val_results["sensitivity_per_class"][i]
               for i, label in enumerate(PATHOLOGY_LABELS)},
            **{f"specificity_{label}": val_results["specificity_per_class"][i]
               for i, label in enumerate(PATHOLOGY_LABELS)},
        })

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_auc: float, tag: str = "best"):
        path = os.path.join(self.save_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "epoch":                epoch,
            "val_auc":              val_auc,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":               self.config,
        }, path)
        print(f"  [Checkpoint] Saved {tag} checkpoint (epoch {epoch}, "
              f"val_auc={val_auc:.4f}) → {path}")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        epochs = self.config["training"]["epochs"]
        print(f"\n[Stage2Trainer] Starting classifier retraining for {epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches:   {len(self.val_loader)}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss  = self._train_epoch(epoch)
            val_results = self._validate()
            val_auc     = val_results["auc_macro"]
            elapsed     = time.time() - t0

            print(f"\nEpoch {epoch}/{epochs} ({elapsed:.0f}s)")
            print(f"  train_loss={train_loss:.4f} | "
                  f"val_loss={val_results['loss']:.4f}")
            print(f"  val_macro_auc={val_auc:.4f} | "
                  f"val_macro_f1={val_results['f1_macro']:.4f} | "
                  f"val_accuracy={val_results['accuracy']:.4f}")
            print(f"  val_macro_sensitivity={val_results['sensitivity_macro']:.4f} | "
                  f"val_macro_specificity={val_results['specificity_macro']:.4f}")

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

            self._save_metrics(epoch, train_loss, val_results)
            self.scheduler.step(val_auc)

            if val_auc > self.best_val_auc:
                self.best_val_auc      = val_auc
                self.epochs_no_improve = 0
                if self.config["output"]["save_best"]:
                    self._save_checkpoint(epoch, val_auc, tag="best")
            else:
                self.epochs_no_improve += 1
                print(f"  [EarlyStopping] No improvement for "
                      f"{self.epochs_no_improve}/{self.early_stop_patience} epochs")
                if self.epochs_no_improve >= self.early_stop_patience:
                    print(f"\n[Stage2Trainer] Early stopping at epoch {epoch}")
                    break

        self._save_checkpoint(epoch, val_auc, tag="final")
        print(f"\n[Stage2Trainer] Done. Best val macro AUC: "
              f"{self.best_val_auc:.4f}")
        wandb.finish()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[Main] Running Stage 2: {config['experiment']['name']}")

    trainer = Stage2Trainer(config)
    trainer.train()