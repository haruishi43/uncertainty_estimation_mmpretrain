#!/usr/bin/env python3

from .edl_classification_losses import (
    EDLSSELoss,
    EDLNLLLoss,
    EDLCELoss,
    EDLKLDivLoss,
)
from .redl_losses import RelaxedEDLSSELoss

__all__ = [
    "EDLSSELoss",
    "EDLNLLLoss",
    "EDLCELoss",
    "EDLKLDivLoss",
    "RelaxedEDLSSELoss",
]
