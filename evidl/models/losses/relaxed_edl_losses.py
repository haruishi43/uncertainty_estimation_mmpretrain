#!/usr/bin/env python3

"""Implementations of R-EDL losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from .orig_edl_losses import kl_div_reg


def relaxed_mse_loss(
    evidence: torch.Tensor,
    y: torch.Tensor,
    lamb1: float = 1.0,
    lamb2: float = 1.0,
) -> torch.Tensor:
    """Relaxed MSE loss from R-EDL.

    `lamb1` is set to 1.0 in the original implementation.
    `lamb2` is flexible in the range of 0.1 to 1.0 and 0.1 was chosen
    in the original implementation.
    """
    num_classes = evidence.shape[-1]

    sum_evidence = torch.sum(evidence, dim=-1, keepdim=True)
    denom = evidence + lamb1 * (sum_evidence - evidence) + lamb2 * num_classes

    gap = y - (evidence + lamb2) / denom

    loss_mse = gap.pow(2).sum(-1)
    return loss_mse.mean()


@MODELS.register_module()
class RelaxedDirichletMSELoss(nn.Module):
    """Dirichlet MSE Loss for R-EDL."""

    def __init__(self, loss_weight: float = 1.0, lamb: int = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.d_lamb = lamb

    def forward(
        self,
        evidence: torch.Tensor,
        label: torch.Tensor,
        step: int,
        max_steps: int,
        lamb: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        # FIXME: for now we don't have `weight` and custom `reduction`

        # lower kl divergence regularization term during the initial training phase
        kl_weight = min(1, float(step) / max_steps)

        num_classes = evidence.shape[-1]
        y = F.one_hot(label, num_classes)
        alpha = evidence + lamb

        mse = relaxed_mse_loss(evidence, y, lamb1=self.d_lamb, lamb2=lamb)
        kl = kl_div_reg(alpha, y)

        return mse + kl_weight * kl
