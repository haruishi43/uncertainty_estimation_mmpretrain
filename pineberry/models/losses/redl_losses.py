"""Implementations of R-EDL losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS


def relaxed_edl_sse_loss(
    evidence: torch.Tensor,
    y: torch.Tensor,
    lamb1: float = 1.0,
    lamb2: float = 1.0,
) -> torch.Tensor:
    """Relaxed SSE loss from R-EDL.

    `lamb1` is set to 1.0 in the original implementation.
    `lamb2` is flexible in the range of 0.1 to 1.0 and 0.1 was chosen
    in the original implementation.
    """
    num_classes = evidence.shape[-1]

    sum_evidence = torch.sum(evidence, dim=-1, keepdim=True)
    denom = evidence + lamb1 * (sum_evidence - evidence) + lamb2 * num_classes

    gap = y - (evidence + lamb2) / denom

    loss_sse = gap.pow(2).sum(-1)
    return loss_sse.mean()


@MODELS.register_module()
class RelaxedEDLSSELoss(nn.Module):
    """SSE Loss for R-EDL."""

    def __init__(
        self,
        lamb: int = 1.0,
        loss_weight: float = 1.0,
        loss_name: str = "redl_sse_loss",
    ) -> None:
        super().__init__()
        self.d_lamb = lamb
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(
        self,
        evidence: torch.Tensor,
        label: torch.Tensor,
        lamb: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        # FIXME: for now we don't have `weight` and custom `reduction`
        num_classes = evidence.shape[-1]
        y = F.one_hot(label, num_classes)
        sse = relaxed_edl_sse_loss(evidence, y, lamb1=self.d_lamb, lamb2=lamb)
        return self.loss_weight * sse

    @property
    def loss_name(self):
        return self._loss_name
