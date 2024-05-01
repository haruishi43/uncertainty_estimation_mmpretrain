#!/usr/bin/env python3

from .orig_edl_losses import (
    DirichletMSELoss,
    DirichletNLLLoss,
    DirichletDigammaLoss,
)
from .relaxed_edl_losses import RelaxedDirichletMSELoss

__all__ = [
    "DirichletMSELoss",
    "DirichletNLLLoss",
    "DirichletDigammaLoss",
    "RelaxedDirichletMSELoss",
]
