#!/usr/bin/env python3

"""Disagreement Metric

Estimate the confidence of an ensemble of estimators.
"""

from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


@METRICS.register_module()
class Disagreement(BaseMetric):

    default_prefix: Optional[str] = "ens"

    def _compute_disagreement(self, preds: Tensor) -> Tensor:
        num_estimators = preds.size(-1)
        counts = torch.sum(F.one_hot(preds), dim=1)
        max_counts = num_estimators * (num_estimators - 1) / 2
        return 1 - (counts * (counts - 1) / 2).sum(dim=1) / max_counts

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            # obtain all prediction score (all ensemble members)
            # shape (num_ensemble, num_classes)
            preds = data_sample["pred_score_all"].clone()
            preds = preds.argmax(dim=-1)

            disagreement = self._compute_disagreement(preds)

            result["disagreement"] = disagreement
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        disagreement = torch.stack([res["disagreement"] for res in results])
        # mean
        disagreement = disagreement.sum(-1) / disagreement.size(-1)
        return dict(disagreement=disagreement.item())
