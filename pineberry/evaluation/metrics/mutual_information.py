#!/usr/bin/env python3

"""Mutual Information.

Estimate the epistemic uncertainty of estimators.
"""

from typing import Any, List, Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


@METRICS.register_module()
class MutualInformation(BaseMetric):

    default_prefix: Optional[str] = "ens"

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            # obtain all prediction score (all ensemble members)
            # shape (num_ensemble, num_classes)
            preds = data_sample["pred_score_all"].clone()
            pred = preds.mean(dim=0)
            entropy_mean = torch.special.entr(pred).sum(dim=-1)
            mean_entropy = torch.special.entr(preds).sum(dim=-1).mean(dim=0)
            mutual_information = entropy_mean - mean_entropy

            result["mi"] = mutual_information
            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        mi = torch.stack([res["mi"] for res in results])
        # mean
        mi = mi.sum(-1) / mi.size(-1)
        return dict(MI=mi.item())
