import torch
import torch.nn as nn


class RobustAsymmetricLoss(nn.Module):
    """
    Robust Asymmetric Loss (RAL) — Park et al., ICCVW 2023.
    Exact port of Ralloss from https://github.com/kalelpark/RAL/blob/main/models/get_optimizer.py
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        lamb: float = 1.5,
        epsilon_neg: float = 0.0,
        epsilon_pos: float = 1.0,
        epsilon_pos_pow: float = -2.5,
        disable_torch_grad_focal_loss: bool = False,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.lamb = lamb
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        # Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric clipping: shift negatives up so easy negatives are harder
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Taylor-expanded positive loss + asymmetric negative loss
        los_pos = y * (
            torch.log(xs_pos.clamp(min=self.eps))
            + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps))
            + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2)
        )
        los_neg = (1 - y) * (
            torch.log(xs_neg.clamp(min=self.eps))
            + self.epsilon_neg * xs_neg.clamp(min=self.eps)
        ) * (self.lamb - x_sigmoid) * x_sigmoid ** 2 * (self.lamb - xs_neg)

        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss = loss * one_sided_w

        return -loss.sum()


def get_ral_loss(**kwargs) -> RobustAsymmetricLoss:
    return RobustAsymmetricLoss(**kwargs)