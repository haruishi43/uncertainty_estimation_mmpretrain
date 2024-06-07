#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from evidl.models.utils import get_curr_iter_info


def edl_nll_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative Log Likelihood Loss.

    Eq. (3) from https://arxiv.org/abs/1806.01768
    """
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    value = torch.sum(
        y * (torch.log(sum_alpha) - torch.log(alpha)),
        dim=1,
        keepdim=True,
    )
    return value.mean()


def edl_ce_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-Entropy Loss with Dirichlet Distribution.

    Eq. (4) from https://arxiv.org/abs/1806.01768
    """
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    value = torch.sum(
        y * (torch.digamma(sum_alpha) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )
    return value.mean()


def edl_sse_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of Squares Loss.

    Eq. (5) from https://arxiv.org/abs/1806.01768
    """
    sum_alpha = alpha.sum(-1, keepdim=True)
    p = alpha / sum_alpha

    # supervision (sse) term
    t1 = (y - p).pow(2).sum(-1, keepdim=True)

    # variance term
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1, keepdim=True)

    loss = t1 + t2
    # print(loss.shape, loss.mean().shape)
    return loss.mean()


def kl_div_reg(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """KL Divergence Regularization.

    KL div of predicted parameters from unifrom Dirichlet distribution.
    """

    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1, keepdim=True)
    sum_beta = beta.sum(-1, keepdim=True)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1, keepdim=True)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1, keepdim=True)
    return kl.mean()


@MODELS.register_module()
class EDLSSELoss(nn.Module):
    """SSE Loss for EDL."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        evidence: torch.Tensor,
        label: torch.Tensor,
        lamb: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        # FIXME: for now we don't have `weight` and custom `reduction`

        # lower kl divergence regularization term during the initial training phase
        step, max_steps = get_curr_iter_info(epoch=True)
        kl_weight = min(1, float(step) / max_steps)
        kl_weight = min(1, float(step) / 10)
        num_classes = evidence.shape[-1]

        alpha = evidence + lamb
        y = F.one_hot(label, num_classes)
        return edl_sse_loss(alpha, y) + kl_weight * kl_div_reg(alpha, y)


@MODELS.register_module()
class EDLNLLLoss(nn.Module):
    """NLL Loss for EDL."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        evidence: torch.Tensor,
        label: torch.Tensor,
        lamb: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        # FIXME: for now we don't have `weight` and custom `reduction`

        step, max_steps = get_curr_iter_info()
        num_classes = evidence.shape[-1]
        kl_weight = min(1, float(step) / max_steps)

        alpha = evidence + lamb
        y = F.one_hot(label, num_classes)
        return edl_nll_loss(alpha, y) + kl_weight * kl_div_reg(alpha, y)


@MODELS.register_module()
class EDLCELoss(nn.Module):
    """Cross-Entropy Loss for EDL."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        evidence: torch.Tensor,
        label: torch.Tensor,
        lamb: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        # FIXME: for now we don't have `weight` and custom `reduction`
        step, max_steps = get_curr_iter_info()
        kl_weight = min(1, float(step) / max_steps)
        num_classes = evidence.shape[-1]

        alpha = evidence + lamb
        y = F.one_hot(label, num_classes)
        return edl_ce_loss(alpha, y) + kl_weight * kl_div_reg(alpha, y)
