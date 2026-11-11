"""Checks the vectorised L_MSC against a slow loop that follows the paper's equations
term by term. Run from the project root:  python tests/test_msc_loss.py"""
import os, sys, math, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from losses.msc_loss import MultiLabelSupConLoss


def naive(z, y, protos, tau, beta, use_p):
    z = F.normalize(z, dim=1); c = F.normalize(protos, dim=1) if use_p else None
    B, L = y.shape; losses = []
    for i in range(B):
        labels = [j for j in range(L) if y[i, j] > 0]
        if not labels:
            continue
        den = sum(beta * math.exp(z[i] @ z[k] / tau) for k in range(B) if k != i)
        if use_p:
            den += sum(math.exp(z[i] @ c[k] / tau) for k in range(L))
        li = 0.0
        for j in labels:
            terms = []
            for l in range(B):
                if l != i and y[l, j] > 0:
                    union = ((y[i] + y[l]) > 0).sum().item()
                    terms.append((1 / union, math.log(math.exp(z[i] @ z[l] / tau) / den)))
            if use_p:
                terms.append((1.0, math.log(math.exp(z[i] @ c[j] / tau) / den)))
            N = sum(f for f, _ in terms)
            if N > 0:
                li += sum(f * lp for f, lp in terms) / N
        losses.append(-li / len(labels))
    return sum(losses) / len(losses)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, d = 12, 15, 8
    z = torch.randn(B, d, dtype=torch.float64)
    y = (torch.rand(B, L) < 0.15).double(); y[0] = 0; y[1, 3] = 1   # one unlabelled anchor
    for use_p, beta in [(True, 0.1), (False, 1.0), (True, 1.0), (False, 0.3)]:
        m = MultiLabelSupConLoss(L, d, temperature=0.1, beta=beta, use_prototypes=use_p).double()
        p = m.prototypes.detach() if use_p else None
        fast, slow = m(z, y).item(), naive(z, y, p, 0.1, beta, use_p)
        assert abs(fast - slow) < 1e-8, (use_p, beta, fast, slow)
        print(f"prototypes={use_p!s:5} beta={beta}: vectorised={fast:.6f} loop={slow:.6f}  OK")
    print("All checks passed.")
