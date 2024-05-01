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

    step: int
    max_steps: int
    step_frac: float

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def set_current_step_from_iter(self, iter: int, max_iters: int) -> None:
        """Set current step from iterations.

        This is called from hooks like `PassStepInfoHook`.
        Mainly used to update loss weights.
        """

        # directly use iter and max_iters from runner
        self.step = iter
        self.max_steps = max_iters
        self.step_frac = iter / max_iters

    def set_current_step_from_epoch(self, epoch: int, max_epochs: int) -> None:
        """Set current step from epochs.

        This is called from hooks like `PassStepInfoHook`.
        Mainly used to update loss weights.
        """

        # directly use epoch and max_epochs from runner
        self.step = epoch
        self.max_steps = max_epochs
        self.step_frac = epoch / max_epochs

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample]) -> dict:
        feats = self.extract_feat(inputs)
        return self.head.loss(
            feats,
            data_samples,
            step=self.step,
            max_steps=self.max_steps,
        )
