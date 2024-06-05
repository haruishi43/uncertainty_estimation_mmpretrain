#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine import MessageHub
from mmpretrain.registry import MODELS


def dirichlet_nll_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


def dirichlet_ce_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


def dirichlet_sse_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of Squares Loss.

    Eq. (5) from https://arxiv.org/abs/1806.01768
    """
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha

    # supervision (sse) term
    t1 = (y - p).pow(2).sum(-1)

    # variance term
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)

    loss = t1 + t2
    return loss.mean()


def kl_div_reg(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """KL Divergence Regularization.

    KL div of predicted parameters from unifrom Dirichlet distribution.
    """

    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


def get_curr_iter_info():
    """Get current iteration and max iterations from the message hub."""
    message_hub = MessageHub.get_current_instance()
    curr_iter = message_hub.get_info("iter")
    max_iters = message_hub.get_info("max_iters")
    return curr_iter, max_iters


@MODELS.register_module()
class DirichletSSELoss(nn.Module):
    """Dirichlet SSELoss."""

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

        # lower kl divergence regularization term during the initial training phase
        step, max_steps = get_curr_iter_info()
        kl_weight = min(1, float(step) / max_steps)
        num_classes = evidence.shape[-1]

        alpha = evidence + lamb
        y = F.one_hot(label, num_classes)
        return dirichlet_sse_loss(alpha, y) + kl_weight * kl_div_reg(alpha, y)


@MODELS.register_module()
class DirichletNLLLoss(nn.Module):
    """Dirichlet NLL Loss."""

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
        return dirichlet_nll_loss(alpha, y) + kl_weight * kl_div_reg(alpha, y)


@MODELS.register_module()
class DirichletCELoss(nn.Module):
    """Dirichlet Cross-Entropy Loss."""

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
        return dirichlet_ce_loss(alpha, y) + kl_weight * kl_div_reg(alpha, y)
