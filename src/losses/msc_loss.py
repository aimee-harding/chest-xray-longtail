"""
Balanced Multi-label Supervised Contrastive loss (L_MSC), without the memory queue.

Audibert et al. (2024), "Exploring Contrastive Learning for Long-Tailed
Multi-Label Text Classification", Section 3.3, Eq. (1)-(5).

    L_MSC = 1/|B| * sum_i l(z_i)

    l(z_i) = -1/|y_i| * sum_{j in Delta(z_i)} 1/N(i,j) * sum_{z_l in P(j,i) U {c_j}}
                 f(z_i, z_l) * log( exp(z_i.z_l / tau) /
                                    sum_{z_k in H U C \\ z_i} g_i(z_k, beta) exp(z_i.z_k / tau) )

    f(z_i, z_l) = 1                    if z_l is a prototype
                  1 / |y_i U y_l|      otherwise                         (Eq. 4)
    N(i,j)      = sum_{z_l in P(j,i) U {c_j}} f(z_i, z_l)                 (Eq. 5)
    g_i(z_k,b)  = 1 if z_k is a prototype, beta otherwise                (Eq. 1)

    Delta(z_i)  = labels of example i
    P(j,i)      = examples (other than i) that have label j
    H           = batch U queue   -> here H = batch only (queue omitted)
    C           = L trainable label prototypes

Because the outer sum runs over each positive label j, a pair sharing m labels
contributes m * 1/|y_i U y_l| = Jaccard similarity in total (paper, after Eq. 5).

Setting use_prototypes=False and beta=1.0 gives the paper's L_Base baseline
(Jaccard-weighted SupCon), which is useful as a sanity check / ablation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelSupConLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        embed_dim: int = 256,
        temperature: float = 0.1,   # paper: 0.1 gave the best results
        beta: float = 0.1,          # paper: beta = 0.1 (not searched)
        use_prototypes: bool = True,
    ):
        super().__init__()
        assert 0.0 < beta <= 1.0
        self.num_labels = num_labels
        self.temperature = temperature
        self.beta = beta
        self.use_prototypes = use_prototypes
        if use_prototypes:
            # L trainable label prototypes C = {c_1, ..., c_L} in the projection space
            self.prototypes = nn.Parameter(torch.randn(num_labels, embed_dim))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        z : (B, d) projected embeddings (normalised inside -> cosine similarity)
        y : (B, L) multi-hot labels (float or bool)
        """
        B = z.shape[0]
        y = y.to(z.dtype)
        z = F.normalize(z, dim=1)
        tau = self.temperature
        eye = torch.eye(B, dtype=torch.bool, device=z.device)

        # ── similarities ─────────────────────────────────────────────────
        sim_inst = (z @ z.t()) / tau                                   # (B, B)
        if self.use_prototypes:
            c = F.normalize(self.prototypes, dim=1)
            sim_proto = (z @ c.t()) / tau                              # (B, L)

        # ── denominator: sum over H U C \ z_i, weighted by g_i ───────────
        # log( beta * exp(s) ) = s + log(beta); the anchor itself is excluded
        log_beta = math.log(self.beta)
        den_inst = (sim_inst + log_beta).masked_fill(eye, float("-inf"))
        if self.use_prototypes:
            log_den = torch.logsumexp(torch.cat([den_inst, sim_proto], dim=1), dim=1)   # (B,)
        else:
            log_den = torch.logsumexp(den_inst, dim=1)

        log_prob_inst = sim_inst - log_den[:, None]                    # (B, B)

        # ── pair weights f(z_i, z_l) = 1 / |y_i U y_l| (instances) ────────
        inter = y @ y.t()                                              # |y_i ∩ y_l|
        card = y.sum(dim=1)
        union = card[:, None] + card[None, :] - inter                  # |y_i U y_l|
        f_inst = (1.0 / union.clamp(min=1.0)).masked_fill(eye, 0.0)

        # For each anchor i and label j:
        #   S[i,j] = sum_{l != i, y_lj = 1} f_il * log_prob_il   (+ prototype term)
        #   N[i,j] = sum_{l != i, y_lj = 1} f_il                 (+ 1 for prototype)
        S = (f_inst * log_prob_inst) @ y                               # (B, L)
        N = f_inst @ y                                                 # (B, L)
        if self.use_prototypes:
            S = S + (sim_proto - log_den[:, None])                     # f = 1 for c_j
            N = N + 1.0

        # only labels the anchor actually has, and that have at least one positive
        valid = (y > 0) & (N > 0)
        term = torch.where(valid, S / N.clamp(min=1e-12), torch.zeros_like(S))

        # l(z_i) = -1/|y_i| * sum_j term[i,j]; anchors without labels are skipped
        has_label = card > 0
        if not has_label.any():
            return z.sum() * 0.0
        loss_i = -term.sum(dim=1)[has_label] / card[has_label]
        return loss_i.mean()


class ProjectionHead(nn.Module):
    """SupCon-style head used in the paper (Appendix A): W2 . ReLU(W1 . x),
    W1 in R^{h x h}, W2 in R^{d x h}, with d = 256. Discarded after Stage 1."""
    def __init__(self, in_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
