#!/usr/bin/env python3

from .orig_edl_losses import (
    DirichletSSELoss,
    DirichletNLLLoss,
    DirichletDigammaLoss,
)
from .relaxed_edl_losses import RelaxedDirichletSSELoss

__all__ = [
    "DirichletSSELoss",
    "DirichletNLLLoss",
    "DirichletDigammaLoss",
    "RelaxedDirichletSSELoss",
]
