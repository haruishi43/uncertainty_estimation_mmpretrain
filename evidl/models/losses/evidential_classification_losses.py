#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


def dirichlet_nll_loss(alpha, y):
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


def dirichlet_digamma_loss(alpha, y):
    """Digamma Loss.

    Eq. (4) from https://arxiv.org/abs/1806.01768
    """
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    value = torch.sum(
        y * (torch.digamma(sum_alpha) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )
    return value.mean()


def dirichlet_mse_loss(alpha, y):
    """Sum of Squares Loss.

    Eq. (5) from https://arxiv.org/abs/1806.01768
    """
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


def kl_div_reg(alpha, y):
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


def evidential_classification(
    alpha: torch.Tensor, y: torch.Tensor, lamb: float = 1.0
) -> torch.Tensor:
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    return dirichlet_mse_loss(alpha, y) + lamb * kl_div_reg(alpha, y)


@MODELS.register_module()
class DirichletMSELoss(nn.Module):
    """Dirichlet MSELoss."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, alpha, label, step: int, max_steps: int, **kwargs):
        # FIXME: for now we don't have `weight` and custom `reduction`

        # lower kl divergence regularization term during the initial training phase
        lamb = min(1, float(step) / max_steps)

        return evidential_classification(alpha, label, lamb=lamb)


@MODELS.register_module()
class DirichletNLLLoss(nn.Module):
    """Dirichlet NLL Loss."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, alpha, label, step: int, max_steps: int, **kwargs):
        # FIXME: for now we don't have `weight` and custom `reduction`

        lamb = min(1, float(step) / max_steps)

        num_classes = alpha.shape[-1]
        y = F.one_hot(y, num_classes)
        return dirichlet_mse_loss(alpha, y) + lamb * kl_div_reg(alpha, y)


@MODELS.register_module()
class DirichletDigammaLoss(nn.Module):
    """Dirichlet Digamma Loss."""

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, alpha, label, step: int, max_steps: int, **kwargs):
        # FIXME: for now we don't have `weight` and custom `reduction`

        lamb = min(1, float(step) / max_steps)

        num_classes = alpha.shape[-1]
        y = F.one_hot(y, num_classes)
        return dirichlet_digamma_loss(alpha, y) + lamb * kl_div_reg(alpha, y)
