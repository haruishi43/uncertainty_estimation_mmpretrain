#!/usr/bin/env python3

"""Implementations of R-EDL losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


def relaxed_mse_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    lamb1: float = 1.0,
    lamb2: float = 1.0,
) -> torch.Tensor:
    """Relaxed MSE loss from R-EDL."""
    num_classes = alpha.shape[-1]

    # evidence
    e = alpha - 1

    gap = y - (e + lamb2) / (
        e + lamb1 * (torch.sum(e, dim=-1, keepdim=True) - e) + lamb2 * num_classes
    )

    loss_mse = gap.pow(2).sum(-1)
    return loss_mse.mean()


def kl_div_reg(alphas, target_alphas):
    """Based on R-EDL implementation.

    This should be the same as the original implementation...
    """

    epsilon = torch.tensor(1e-8)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(
        torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term)
    )
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(
        torch.lgamma(target_alphas + epsilon)
        - torch.lgamma(alphas + epsilon)
        + (alphas - target_alphas)
        * (torch.digamma(alphas + epsilon) - torch.digamma(alp0 + epsilon)),
        dim=-1,
        keepdim=True,
    )
    alphas_term = torch.where(
        torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term)
    )
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = torch.squeeze(alp0_term + alphas_term).mean()

    return loss


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

        num_classes = alpha.shape[-1]
        y = F.one_hot(y, num_classes)
        return relaxed_mse_loss(alpha, y) + lamb * kl_div_reg(alpha, y)
