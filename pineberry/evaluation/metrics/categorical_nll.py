#!/usr/bin/env python3

"""Categorical NLL.

Accuracy + Confidence
"""

from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F

from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS


@METRICS.register_module()
class CategoricalNLL(BaseMetric):
    """Negative Log Likelihood Metric."""

    default_prefix: Optional[str] = "accuracy"

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            pred = data_sample["pred_score"].clone()
            num_classes = pred.size()[-1]

            # not one-hot
            if "gt_score" in data_sample:
                target = data_sample["gt_score"].clone()
                if target.ndim == pred.ndim:
                    target = target.argmax(dim=1)
            else:
                target = data_sample["gt_label"].clone()

            nll = F.nll_loss(torch.log(pred).unsqueeze(0), target, reduction="sum")

            result["nll"] = nll
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        nll = torch.stack([res["nll"] for res in results])
        # mean
        nll = nll.sum(-1) / nll.size(-1)
        return dict(NLL=nll.item())
