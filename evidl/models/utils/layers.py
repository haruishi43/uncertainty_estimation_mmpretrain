#!/usr/bin/env python3

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


class Dirichlet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.fc(x)
        alpha = self.evidence(out) + 1
        return alpha
