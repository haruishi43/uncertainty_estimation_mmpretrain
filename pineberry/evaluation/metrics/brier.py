#!/usr/bin/env python3

"""Brier Score."""

from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.structures import label_to_onehot
from mmpretrain.registry import METRICS


@METRICS.register_module()
class BrierScore(BaseMetric):
    """The Brier Score Metric.

    Args:
        top_class (bool, optional): If true, compute the Brier score for the
            top class only. Defaults to False.
    """

    default_prefix: Optional[str] = "calib_err"

    def __init__(
        self,
        top_class: bool = False,
    ) -> None:
        super().__init__()
        self.top_class = top_class

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            pred = data_sample["pred_score"].clone()
            num_classes = pred.size()[-1]

            if "gt_score" in data_sample:
                target = data_sample["gt_score"].clone()
            else:
                target = data_sample["gt_label"].clone()
            target = label_to_onehot(target, num_classes)

            if self.top_class:
                prob, index = pred.max(dim=-1)
                target = target.gather(-1, index.unsqueeze(-1)).squeeze(-1)
                brier_score = F.mse_loss(prob, target, reduction="none")
            else:
                brier_score = F.mse_loss(pred, target, reduction="none").sum(dim=-1)

            result["brier"] = brier_score

            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        brier = torch.stack([res["brier"] for res in results])
        # mean
        brier = brier.sum(-1) / brier.size(-1)
        return dict(brier=brier.item())
