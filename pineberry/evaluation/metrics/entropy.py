#!/usr/bin/env python3

"""Shannon Entropy Metric."""

from typing import Any, List, Optional, Sequence

import torch

from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS


@METRICS.register_module()
class Entropy(BaseMetric):
    """Shannon Entropy Metric.

    Estimate the confidence of a single model or the mean confidence across estimators.
    """

    default_prefix: Optional[str] = "uncert"

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            pred = data_sample["pred_score"].clone()
            entropy = torch.special.entr(pred).sum(dim=-1)

            result["entropy"] = entropy
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        entropy = torch.stack([res["entropy"] for res in results])
        # mean
        entropy = entropy.sum(-1) / entropy.size(-1)
        return dict(entropy=entropy.item())
