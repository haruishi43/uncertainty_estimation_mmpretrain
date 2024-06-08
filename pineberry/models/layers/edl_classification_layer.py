#!/usr/bin/env python3

from functools import partial
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmpretrain.registry import MODELS


def _clip_exp(x: torch.Tensor, clip_value: float = 10.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, min=-clip_value, max=clip_value))


def _exp_tanh(x: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Exponential tanh function introduced in image caption retrieval.

    `tau` is a scaling factor between 0 and 1, and is usually set to
    1/K where K is the number of classes.
    """
    return torch.exp(torch.tanh(x) / tau)


@MODELS.register_module()
class DirichletLayer(BaseModule):
    """Dirichlet Layer for EDL.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        evidence_function (str): Evidence function. Defaults to 'softplus'.
        clip_value (float): Value to clip the exponential function. Defaults to 10.0.
        dropout_ratio (float): Dropout ratio. Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        evidence_function: str = "softplus",
        clip_value: float = 10.0,
        dropout_ratio: float = 0.5,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        if dropout_ratio:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Linear(in_channels, out_channels)
        self.num_classes = out_channels

        # evidence functions
        if evidence_function == "softplus":
            self.func_evidence = F.softplus
        elif evidence_function == "exp":
            self.func_evidence = partial(_clip_exp, clip_value=clip_value)
        elif evidence_function == "relu":
            self.func_evidence = F.relu
        elif evidence_function == "exp_tanh":
            self.func_evidence = partial(_exp_tanh, tau=0.25)
        else:
            raise ValueError(f"Unknown evidence function: {evidence_function}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        if self.dropout:
            x = self.dropout(x)
        out = self.fc(x)

        # The main functionality of the model is to calculate the evidence.
        # Calculating alpha and other outputs should be done elsewhere.
        evidence = self.func_evidence(out)

        return evidence
