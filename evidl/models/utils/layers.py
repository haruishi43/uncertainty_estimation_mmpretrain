#!/usr/bin/env python3

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta


def _clip_exp(x: torch.Tensor, clip_value: float = 10.0):
    return torch.exp(torch.clamp(x, min=-clip_value, max=clip_value))


def _exp_tanh(x: torch.Tensor, tau: float = 0.1):
    """Exponential tanh function introduced in image caption retrieval.

    `tau` is a scaling factor between 0 and 1, and is usually set to
    1/K where K is the number of classes.
    """
    return torch.exp(torch.tanh(x) / tau)


class Dirichlet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        evidence_function: str = "softplus",
        clip_value: float = 10.0,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes

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

    def forward(self, x):
        out = self.fc(x)

        # The main functionality of the model is to calculate the evidence.
        # Calculating alpha and other outputs should be done elsewhere.
        evidence = self.func_evidence(out)
        return evidence
