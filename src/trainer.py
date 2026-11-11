
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import roc_auc_score
import wandb

import sys
sys.path.append('/home-mscluster/aharding/chest_xray_longtail/src')

from datasets.chest_xray_dataset import (
    ChestXray14Dataset,
    PATHOLOGY_LABELS,
    NUM_CLASSES,
)
from models.densenet import get_backbone, freeze_backbone, unfreeze_backbone


# config loading 

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# loss functions

def get_loss_fn(config: dict, pos_weight: torch.Tensor = None):

    loss_cfg = config["training"]["loss"]

    if loss_cfg["type"] == "bce":
        if loss_cfg.get("use_pos_weight", False) and pos_weight is not None:
            print(f"[Loss] BCE with pos_weight (min={pos_weight.min():.1f}, "
                  f"max={pos_weight.max():.1f})")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print("[Loss] Plain BCE (no pos_weight)")
            return nn.BCEWithLogitsLoss()

    elif loss_cfg["type"] == "ral":
        from losses.ral_loss import RobustAsymmetricLoss
        print("[Loss] Robust Asymmetric Loss (RAL)")
        return RobustAsymmetricLoss(
            gamma_neg=loss_cfg.get("gamma_neg", 4.0),
            gamma_pos=loss_cfg.get("gamma_pos", 0.0),
            clip=loss_cfg.get("clip", 0.05),
            lamb=loss_cfg.get("lamb", 1.5),
            epsilon_pos=loss_cfg.get("epsilon_pos", 1.0),
            epsilon_pos_pow=loss_cfg.get("epsilon_pos_pow", -2.5),
            epsilon_neg=loss_cfg.get("epsilon_neg", 0.0),
        )

    raise ValueError(f"Unknown loss type: {loss_cfg['type']}")

# optimiser and scheduler 

def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    opt_cfg = config["training"]["optimizer"]
    lr = opt_cfg["lr"]
    wd = opt_cfg.get("weight_decay", 0.0)

    # only optimise parameters that require gradients (non frozen)
    params = filter(lambda p: p.requires_grad, model.parameters())

    if opt_cfg["type"] == "adam":
        print(f"[Optimiser] Adam lr={lr}, weight_decay={wd}")
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)

    raise ValueError(f"Unknown optimiser: {opt_cfg['type']}")


def get_scheduler(optimizer, config: dict):
    sched_cfg = config["training"]["scheduler"]

    if sched_cfg["type"] == "plateau":
        # reduce LR when val macro AUC stops improving
        # 'max' mode because higher AUC is better
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=sched_cfg["patience"],
            factor=sched_cfg["factor"],
            min_lr=sched_cfg["min_lr"],
            verbose=True,
        )

    raise ValueError(f"Unknown scheduler: {sched_cfg['type']}")


# evaluation 

def evaluate(all_labels: np.ndarray, all_probs: np.ndarray, threshold: float = 0.5) -> dict:
    #all labels: binary ground truths
    #all probs: predicted probabilities 

    # binarise predictions at threshold for threshold-dependent metrics
    preds = (all_probs >= threshold).astype(np.float32)  

    auc_per_class = []
    f1_per_class = []
    sensitivity_per_class = []
    specificity_per_class = []

    for c in range(NUM_CLASSES):
        y_true = all_labels[:, c]
        y_prob = all_probs[:, c]
        y_pred = preds[:, c]

        n_pos = y_true.sum()
        n_neg = (1 - y_true).sum()

        # AUC 
        # requires at least one positive example
        if n_pos == 0:
            auc_per_class.append(float("nan"))
        else:
            auc_per_class.append(roc_auc_score(y_true, y_prob))

        # Confusion matrix components 
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        # F1 
        # F1 = 2*TP / (2*TP + FP + FN)
        # NaN if no positives and no predicted positives (undefined)
        denom_f1 = 2 * tp + fp + fn
        f1_per_class.append(
            float("nan") if n_pos == 0
            else (2 * tp / denom_f1 if denom_f1 > 0 else 0.0)
        )

        # Sensitivity (recall/true positive rate)
        # Sensitivity = TP / (TP + FN)
        # "Of all images truly containing pathology c, how many did we catch?"
        sensitivity_per_class.append(
            float("nan") if n_pos == 0
            else (tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        )

        # Specificity (true negative rate) 
        # Specificity = TN / (TN + FP)
        # "Of all images truly NOT containing pathology c, how many did we correctly identify as negative?"
        specificity_per_class.append(
            float("nan") if n_neg == 0
            else (tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        )

    # Macro averages (excluding NaN classes)
    def macro(values):
        valid = [v for v in values if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    # Overall accuracy
    accuracy = float((preds == all_labels).mean())

    return {
        "auc_per_class":         auc_per_class,
        "auc_macro":             macro(auc_per_class),
        "f1_per_class":          f1_per_class,
        "f1_macro":              macro(f1_per_class),
        "sensitivity_per_class": sensitivity_per_class,
        "sensitivity_macro":     macro(sensitivity_per_class),
        "specificity_per_class": specificity_per_class,
        "specificity_macro":     macro(specificity_per_class),
        "accuracy":              accuracy,
    }


# Trainer
class Trainer:
    #stage field in config determines behaviour
    #joint: full end-to-end training, no freezing (Config 1)
    #stage1: full end-to-end training, checkpoint is used to initialise Stage 2

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        # Datasets 
        data_cfg = config["data"]
        self.train_ds = ChestXray14Dataset(
            data_cfg["data_dir"], split="train",
            val_frac=data_cfg["val_frac"], seed=data_cfg["seed"],
        )
        self.val_ds = ChestXray14Dataset(
            data_cfg["data_dir"], split="val",
            val_frac=data_cfg["val_frac"], seed=data_cfg["seed"],
        )

        # DataLoaders
        # Stage 1 and joint training both use instance-balanced sampling
        # standard random shuffling
        batch_size  = config["training"]["batch_size"]
        num_workers = data_cfg["num_workers"]

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,           # instance-balanced = random shuffle
            num_workers=num_workers,
            pin_memory=True,        # faster GPU transfer
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,  #don't need random sampling for evaluation, want to evaluate val set systematically
            num_workers=num_workers,
            pin_memory=True,
        )

        # Model 
        model_cfg = config["model"]
        self.model = get_backbone(
            num_classes=model_cfg["num_classes"],
            pretrained=model_cfg["pretrained"],
        ).to(self.device)

        # Both joint and stage1 train everything end-to-end
        unfreeze_backbone(self.model)

        # Loss 
        pos_weight = self.train_ds.get_pos_weight().to(self.device)
        self.criterion = get_loss_fn(config, pos_weight)

        # Optimiser and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        # Output directory 
        self.save_dir = config["output"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        #W&Bs init
        wandb.init(
            project="chest-xray-longtail",
            name=config["experiment"]["name"],
            config={
                "batch_size":   config["training"]["batch_size"],
                "lr":           config["training"]["optimizer"]["lr"],
                "weight_decay": config["training"]["optimizer"]["weight_decay"],
                "loss":         config["training"]["loss"]["type"],
                "epochs":       config["training"]["epochs"],
                "sampling":     config["training"]["sampling"],
                "stage":        config["experiment"]["stage"],
            }
        )

        # Training state
        self.best_val_auc   = 0.0
        self.epochs_no_improve = 0
        self.early_stop_patience = config["training"]["early_stopping"]["patience"]

        # Metrics log 
        self.metrics_log = []
        self.metrics_path = os.path.join(self.save_dir, "metrics.csv")
        self.metrics_json_path = os.path.join(self.save_dir, "metrics.json")

    # Single epoch

    def _train_epoch(self, epoch: int) -> float:
        self.model.train() #model in training mode
        total_loss = 0.0
        n_batches  = len(self.train_loader)

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad() #clears old gradients
            logits = self.model(images)          # forward pass, gives (B, 14)
            loss   = self.criterion(logits, labels)
            loss.backward() #backprop
            self.optimizer.step() #update weights

            total_loss += loss.item()

            # Progress print every 50 batches
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch} [{i+1}/{n_batches}] "
                      f"loss={loss.item():.4f}")

        return total_loss / n_batches

    # Validation 

    @torch.no_grad()  #not updating model during validation, dont track grads
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

        all_labels = np.concatenate(all_labels, axis=0)  # (N, 14)
        all_probs  = np.concatenate(all_probs,  axis=0)  # (N, 14)

        metrics = evaluate(all_labels, all_probs, threshold=0.5)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def _save_metrics(self, epoch: int, train_loss: float, val_results: dict) -> None:
        import csv, json

        # Build flat row for CSV 
        row = {
            "epoch":                epoch,
            "train_loss":           round(train_loss, 6),
            "val_loss":             round(val_results["loss"], 6),
            "auc_macro":            round(val_results["auc_macro"], 6),
            "f1_macro":             round(val_results["f1_macro"], 6),
            "sensitivity_macro":    round(val_results["sensitivity_macro"], 6),
            "specificity_macro":    round(val_results["specificity_macro"], 6),
            "accuracy":             round(val_results["accuracy"], 6),
        }

        # Add per-class columns
        for i, label in enumerate(PATHOLOGY_LABELS):
            row[f"auc_{label}"]         = round(val_results["auc_per_class"][i], 6)
            row[f"f1_{label}"]          = round(val_results["f1_per_class"][i], 6)
            row[f"sensitivity_{label}"] = round(val_results["sensitivity_per_class"][i], 6)
            row[f"specificity_{label}"] = round(val_results["specificity_per_class"][i], 6)

        # Write CSV (append mode, write header only on first epoch) 
        write_header = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Write JSON 
        self.metrics_log.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val": {
                "loss":             val_results["loss"],
                "auc_macro":        val_results["auc_macro"],
                "f1_macro":         val_results["f1_macro"],
                "sensitivity_macro":val_results["sensitivity_macro"],
                "specificity_macro":val_results["specificity_macro"],
                "accuracy":         val_results["accuracy"],
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

        # Log to W&B
        wandb.log({
            "epoch":               epoch,
            "train_loss":          train_loss,
            "val_loss":            val_results["loss"],
            "auc_macro":           val_results["auc_macro"],
            "f1_macro":            val_results["f1_macro"],
            "sensitivity_macro":   val_results["sensitivity_macro"],
            "specificity_macro":   val_results["specificity_macro"],
            "accuracy":            val_results["accuracy"],
            **{f"auc_{label}": val_results["auc_per_class"][i]
            for i, label in enumerate(PATHOLOGY_LABELS)},
            **{f"f1_{label}": val_results["f1_per_class"][i]
            for i, label in enumerate(PATHOLOGY_LABELS)},
            **{f"sensitivity_{label}": val_results["sensitivity_per_class"][i]
            for i, label in enumerate(PATHOLOGY_LABELS)},
            **{f"specificity_{label}": val_results["specificity_per_class"][i]
            for i, label in enumerate(PATHOLOGY_LABELS)},
        })

    # Checkpointing 

    def _save_checkpoint(self, epoch: int, val_auc: float, tag: str = "best"):
        path = os.path.join(self.save_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "epoch":      epoch,
            "val_auc":    val_auc,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":     self.config,
        }, path)
        print(f"  [Checkpoint] Saved {tag} checkpoint (epoch {epoch}, "
              f"val_auc={val_auc:.4f}) → {path}")

    # Main training loop 
    def train(self):
        epochs = self.config["training"]["epochs"]
        print(f"\n[Trainer] Starting training for {epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches:   {len(self.val_loader)}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train
            train_loss = self._train_epoch(epoch)

            # Validate 
            val_results = self._validate()
            val_auc     = val_results["auc_macro"]
            val_loss    = val_results["loss"]

            elapsed = time.time() - t0

            # Log 
            val_auc  = val_results["auc_macro"]
            val_loss = val_results["loss"]

            print(f"\nEpoch {epoch}/{epochs} ({elapsed:.0f}s)")
            print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
            print(f"  val_macro_auc={val_auc:.4f} | "
                  f"val_macro_f1={val_results['f1_macro']:.4f} | "
                  f"val_accuracy={val_results['accuracy']:.4f}")
            print(f"  val_macro_sensitivity={val_results['sensitivity_macro']:.4f} | "
                  f"val_macro_specificity={val_results['specificity_macro']:.4f}")

            # Per-class breakdown 
            TAIL = {"Pneumonia", "Hernia", "Fibrosis"}
            print(f"\n  {'Class':<22} {'AUC':>6} {'F1':>6} "
                  f"{'Sens':>6} {'Spec':>6}")
            print(f"  {'-'*52}")
            for i, label in enumerate(PATHOLOGY_LABELS):
                marker = " *" if label in TAIL else ""
                auc  = val_results["auc_per_class"][i]
                f1   = val_results["f1_per_class"][i]
                sens = val_results["sensitivity_per_class"][i]
                spec = val_results["specificity_per_class"][i]
                print(f"  {label:<22} "
                      f"{auc:>6.4f} {f1:>6.4f} "
                      f"{sens:>6.4f} {spec:>6.4f}{marker}")

            # Save metrics to disk 
            self._save_metrics(epoch, train_loss, val_results)

            # Scheduler step 
            val_auc = val_results["auc_macro"]
            self.scheduler.step(val_auc)

            #  Early stopping + checkpointing 
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_no_improve = 0
                if self.config["output"]["save_best"]:
                    self._save_checkpoint(epoch, val_auc, tag="best")
            else:
                self.epochs_no_improve += 1
                print(f"  [EarlyStopping] No improvement for "
                      f"{self.epochs_no_improve}/{self.early_stop_patience} epochs")

                if self.epochs_no_improve >= self.early_stop_patience:
                    print(f"\n[Trainer] Early stopping triggered at epoch {epoch}")
                    break

        # Save final checkpoint regardless
        self._save_checkpoint(epoch, val_auc, tag="final")
        print(f"\n[Trainer] Training complete. Best val macro AUC: "
              f"{self.best_val_auc:.4f}")
        
        wandb.finish()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[Main] Running experiment: {config['experiment']['name']}")

    trainer = Trainer(config)
    trainer.train()