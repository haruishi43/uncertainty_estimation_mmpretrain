#!/usr/bin/env python3

from typing import List, Optional

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models.classifiers.image import ImageClassifier


@MODELS.register_module()
class EvidentialImageClassifier(ImageClassifier):
    """Evidential Deep Learning Image Classifier."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample]) -> dict:
        feats = self.extract_feat(inputs)
        return self.head.loss(
            feats,
            data_samples,
        )
