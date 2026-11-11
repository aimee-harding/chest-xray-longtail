"""
Stage 1 for Config 5: supervised contrastive representation learning with the
balanced Multi-label Supervised Contrastive loss (L_MSC, Audibert et al. 2024),
without the MoCo memory queue.

Follows the paper's contrastive pre-training recipe (Appendix A) where it applies:
  - backbone trained with the contrastive loss ONLY (no BCE on the backbone)
  - SupCon projection head W2.ReLU(W1.x), d = 256, discarded after Stage 1
  - L trainable label prototypes, beta = 0.1, temperature = 0.1
  - cosine learning-rate schedule, gradient clipping at 1.0
  - no extra data augmentation (the standard train transform is kept, as for
    every other configuration, so the comparison stays controlled)

Project-specific choices (stated in the report):
  - "No Finding" is treated as an extra (15th) label for the contrastive loss only,
    because the paper's loss is undefined for examples without labels.
  - Optimiser, learning rate, weight decay and batch size match Configs 1-4.
  - An "online linear probe" (the model's 14-way classifier) is trained on DETACHED
    features with BCE. It never sends gradients into the backbone; it is only used
    to pick the checkpoint by validation macro AUC, like the paper picks its
    checkpoint by linear evaluation. Because it lives in model.classifier, the saved
    checkpoint loads directly into linear_probe.py, stage2_trainer.py and
    evaluate_test.py without changes.

Usage:
    python src/contrastive_trainer.py --config configs/config5_msc_stage1.yaml
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.chest_xray_dataset import ChestXray14Dataset, PATHOLOGY_LABELS, NUM_CLASSES  # noqa: E402
from models.densenet import get_backbone, unfreeze_backbone  # noqa: E402
from losses.msc_loss import MultiLabelSupConLoss, ProjectionHead  # noqa: E402
from trainer import Trainer, evaluate, load_config  # noqa: E402  (reuse metric logging)


class ContrastiveTrainer(Trainer):
    # NOTE: deliberately does not call Trainer.__init__ (that builds a BCE/RAL setup);
    # only the logging/checkpoint helpers of Trainer are reused.
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ContrastiveTrainer] Using device: {self.device}")

        data_cfg, train_cfg, loss_cfg = config["data"], config["training"], config["training"]["loss"]

        # ── data (instance-balanced, as for every Stage 1) ────────────────
        self.train_ds = ChestXray14Dataset(data_cfg["data_dir"], split="train",
                                           val_frac=data_cfg["val_frac"], seed=data_cfg["seed"])
        self.val_ds = ChestXray14Dataset(data_cfg["data_dir"], split="val",
                                         val_frac=data_cfg["val_frac"], seed=data_cfg["seed"])
        bs, nw = train_cfg["batch_size"], data_cfg["num_workers"]
        # drop_last: a contrastive batch of 1 has no pairs
        self.train_loader = DataLoader(self.train_ds, batch_size=bs, shuffle=True,
                                       num_workers=nw, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)

        # ── model: backbone + projection head + online probe (model.classifier) ──
        self.model = get_backbone(num_classes=config["model"]["num_classes"],
                                  pretrained=config["model"]["pretrained"]).to(self.device)
        unfreeze_backbone(self.model)
        feat_dim = self.model.classifier.in_features            # 1024
        embed_dim = loss_cfg.get("embed_dim", 256)
        self.proj_head = ProjectionHead(feat_dim, embed_dim).to(self.device)

        # ── losses ────────────────────────────────────────────────────────
        self.no_finding_as_label = loss_cfg.get("no_finding_as_label", True)
        n_con_labels = NUM_CLASSES + (1 if self.no_finding_as_label else 0)
        self.criterion = MultiLabelSupConLoss(
            num_labels=n_con_labels, embed_dim=embed_dim,
            temperature=loss_cfg.get("temperature", 0.1),
            beta=loss_cfg.get("beta", 0.1),
            use_prototypes=loss_cfg.get("use_prototypes", True),
        ).to(self.device)
        self.probe_criterion = nn.BCEWithLogitsLoss()
        print(f"[Loss] L_MSC (no queue): tau={self.criterion.temperature}, beta={self.criterion.beta}, "
              f"prototypes={self.criterion.use_prototypes}, contrastive labels={n_con_labels} "
              f"(No Finding as label: {self.no_finding_as_label})")

        # ── optimiser: same settings as Configs 1-4 for the representation ──
        opt_cfg, probe_cfg = train_cfg["optimizer"], train_cfg.get("probe", {})
        lr, wd = opt_cfg["lr"], opt_cfg.get("weight_decay", 0.0)
        self.con_params = list(self.model.features.parameters()) + list(self.proj_head.parameters())
        groups = [{"params": self.con_params, "lr": lr, "weight_decay": wd}]
        if self.criterion.use_prototypes:
            groups.append({"params": [self.criterion.prototypes], "lr": lr, "weight_decay": 0.0})
            self.con_params = self.con_params + [self.criterion.prototypes]
        groups.append({"params": self.model.classifier.parameters(),
                       "lr": probe_cfg.get("lr", 1e-3), "weight_decay": probe_cfg.get("weight_decay", 0.0)})
        if opt_cfg["type"] != "adam":
            raise ValueError("Only 'adam' is supported here (to match Configs 1-4).")
        self.optimizer = torch.optim.Adam(groups)
        print(f"[Optimiser] Adam lr={lr}, weight_decay={wd} (probe lr={probe_cfg.get('lr', 1e-3)})")

        sched_cfg = train_cfg["scheduler"]
        if sched_cfg["type"] != "cosine":
            raise ValueError("Config 5 uses a cosine schedule, as in the paper.")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["epochs"], eta_min=sched_cfg.get("min_lr", 0.0))
        self.grad_clip = train_cfg.get("grad_clip", 1.0)

        # ── output / logging (same files as the other trainers) ───────────
        self.save_dir = config["output"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        wandb.init(project="chest-xray-longtail", name=config["experiment"]["name"], config={
            "batch_size": bs, "lr": lr, "weight_decay": wd, "loss": "msc",
            "temperature": self.criterion.temperature, "beta": self.criterion.beta,
            "epochs": train_cfg["epochs"], "sampling": train_cfg["sampling"],
            "stage": config["experiment"]["stage"]})
        self.best_val_auc = 0.0
        self.best_val_con = float("inf")
        self.epochs_no_improve = 0
        self.early_stop_patience = train_cfg["early_stopping"]["patience"]
        self.metrics_log = []
        self.metrics_path = os.path.join(self.save_dir, "metrics.csv")
        self.metrics_json_path = os.path.join(self.save_dir, "metrics.json")

    # contrastive label vector: 14 pathologies (+ "No Finding")
    def _con_labels(self, labels):
        if not self.no_finding_as_label:
            return labels
        no_finding = (labels.sum(dim=1, keepdim=True) == 0).float()
        return torch.cat([labels, no_finding], dim=1)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train(); self.proj_head.train()
        total_con, total_probe, n = 0.0, 0.0, len(self.train_loader)
        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            feats = self.model.get_features(images)                  # (B, 1024)
            con_loss = self.criterion(self.proj_head(feats), self._con_labels(labels))
            probe_loss = self.probe_criterion(self.model.classifier(feats.detach()), labels)

            (con_loss + probe_loss).backward()                       # probe grads stop at the head
            torch.nn.utils.clip_grad_norm_(self.con_params, self.grad_clip)
            self.optimizer.step()

            total_con += con_loss.item(); total_probe += probe_loss.item()
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch} [{i+1}/{n}] con_loss={con_loss.item():.4f} "
                      f"probe_bce={probe_loss.item():.4f}")
        return total_con / n

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval(); self.proj_head.eval()
        all_labels, all_probs, con_sum, n = [], [], 0.0, 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            feats = self.model.get_features(images)
            if images.shape[0] > 1:
                con_sum += self.criterion(self.proj_head(feats), self._con_labels(labels)).item(); n += 1
            all_probs.append(torch.sigmoid(self.model.classifier(feats)).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        metrics = evaluate(np.concatenate(all_labels), np.concatenate(all_probs), threshold=0.5)
        metrics["loss"] = con_sum / max(n, 1)                         # val contrastive loss
        return metrics

    def _save_checkpoint(self, epoch: int, val_auc: float, tag: str = "best"):
        path = os.path.join(self.save_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "epoch": epoch, "val_auc": val_auc,
            "model_state_dict": self.model.state_dict(),             # backbone + online probe
            "proj_head_state_dict": self.proj_head.state_dict(),
            "contrastive_loss_state_dict": self.criterion.state_dict(),   # prototypes
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        print(f"  [Checkpoint] Saved {tag} checkpoint (epoch {epoch}, val_auc={val_auc:.4f}) → {path}")

    def train(self):
        epochs = self.config["training"]["epochs"]
        print(f"\n[ContrastiveTrainer] Starting training for {epochs} epochs")
        print(f"  Train batches: {len(self.train_loader)}\n  Val batches:   {len(self.val_loader)}\n")
        TAIL = {"Hernia", "Pneumonia", "Fibrosis"}
        epoch = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val = self._validate()
            val_auc = val["auc_macro"]
            print(f"\nEpoch {epoch}/{epochs} ({time.time() - t0:.0f}s)  lr={self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  train_con_loss={train_loss:.4f} | val_con_loss={val['loss']:.4f}")
            print(f"  online probe: val_macro_auc={val_auc:.4f} | val_macro_f1={val['f1_macro']:.4f}")
            print(f"\n  {'Class':<22} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6}\n  {'-'*52}")
            for i, label in enumerate(PATHOLOGY_LABELS):
                print(f"  {label:<22} {val['auc_per_class'][i]:>6.4f} {val['f1_per_class'][i]:>6.4f} "
                      f"{val['sensitivity_per_class'][i]:>6.4f} {val['specificity_per_class'][i]:>6.4f}"
                      f"{' *' if label in TAIL else ''}")

            # metrics.csv: train_loss / val_loss columns hold the CONTRASTIVE losses
            self._save_metrics(epoch, train_loss, val)
            self.scheduler.step()

            if val["loss"] < self.best_val_con:                       # paper's candidate checkpoint
                self.best_val_con = val["loss"]
                self._save_checkpoint(epoch, val_auc, tag="lowest_val_loss")
            if val_auc > self.best_val_auc:                           # used downstream
                self.best_val_auc = val_auc
                self.epochs_no_improve = 0
                if self.config["output"]["save_best"]:
                    self._save_checkpoint(epoch, val_auc, tag="best")
            else:
                self.epochs_no_improve += 1
                print(f"  [EarlyStopping] No improvement for {self.epochs_no_improve}/{self.early_stop_patience} epochs")
                if self.epochs_no_improve >= self.early_stop_patience:
                    print(f"\n[ContrastiveTrainer] Early stopping triggered at epoch {epoch}")
                    break

        self._save_checkpoint(epoch, val_auc, tag="final")
        print(f"\n[ContrastiveTrainer] Done. Best online-probe val macro AUC: {self.best_val_auc:.4f}")
        wandb.finish()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    print(f"[Main] Running experiment: {cfg['experiment']['name']}")
    ContrastiveTrainer(cfg).train()
