#!/usr/bin/env python3

import math
from typing import Any, List, Optional, Sequence

from sklearn.metrics import auc

import torch
from torch import Tensor
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS


def _aurc_rejection_rate_compute(
    scores: Tensor,
    errors: Tensor,
) -> Tensor:
    """Compute the cumulative error rates for a given set of scores and errors.

    Args:
        scores (Tensor): uncertainty scores of shape :math:`(B,)`
        errors (Tensor): binary errors of shape :math:`(B,)`
    """
    num_samples = scores.size(0)
    errors = errors[scores.argsort(descending=True)]
    return errors.cumsum(dim=-1) / torch.arange(
        1, num_samples + 1, dtype=scores.dtype, device=scores.device
    )


@METRICS.register_module()
class AURC(BaseMetric):
    """Area Under the Risk-Coverage curve.

    The Area Under the Risk-Coverage curve (AURC) is the main metric for
    Selective Classification (SC) performance assessment. It evaluates the
    quality of uncertainty estimates by measuring the ability to
    discriminate between correct and incorrect predictions based on their
    rank (and not their values in contrast with calibration).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    Args:
        kwargs: Additional keyword arguments.

    Reference:
        Geifman & El-Yaniv. "Selective classification for deep neural
            networks." In NeurIPS, 2017.
    """

    default_prefix: Optional[str] = "sc"

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            pred = data_sample["pred_score"].clone()
            if pred.ndim == 0:
                # binary
                pred = torch.stack([1 - pred, pred], dim=-1)

            if "gt_score" in data_sample:
                target = data_sample["gt_score"].clone()
                if target.ndim == pred.ndim:
                    target = target.argmax(dim=1)
            else:
                target = data_sample["gt_label"].clone()

            result["score"] = pred.max(dim=-1).values.reshape(1)
            result["error"] = (pred.argmax(dim=-1) != target).float()

            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        scores = torch.cat([res["score"] for res in results])
        errors = torch.cat([res["error"] for res in results])

        error_rates = _aurc_rejection_rate_compute(scores, errors).cpu()
        num_samples = error_rates.size(0)
        x = torch.arange(1, num_samples + 1, device="cpu") / num_samples

        aurc = torch.tensor([auc(x, error_rates)], device="cpu") / (1 - 1 / num_samples)
        return dict(aurc=aurc.item())


def _risk_coverage_checks(threshold: float) -> None:
    if not isinstance(threshold, float):
        raise TypeError(
            f"Expected threshold to be of type float, but got {type(threshold)}"
        )
    if threshold < 0 or threshold > 1:
        raise ValueError(
            f"Threshold should be in the range [0, 1], but got {threshold}."
        )


@METRICS.register_module()
class CovAtxRisk(AURC):
    """Coverage at x Risk.

    If there are multiple coverage values corresponding to the given risk,
    i.e., the risk(coverage) is not monotonic, the coverage at x risk is
    the maximum coverage value corresponding to the given risk. If no
    there is no coverage value corresponding to the given risk, return
    float("nan").

    Args:
        risk_threshold (float): The risk threshold at which to compute the
            coverage.
        kwargs: Additional arguments to pass to the metric class.
    """

    default_prefix: Optional[str] = "sc"

    def __init__(self, risk_threshold: float, **kwargs) -> None:
        super().__init__(**kwargs)
        _risk_coverage_checks(risk_threshold)
        self.risk_threshold = risk_threshold

    def compute_metrics(self, results: List) -> dict:
        scores = torch.cat([res["score"] for res in results])
        errors = torch.cat([res["error"] for res in results])

        error_rates = _aurc_rejection_rate_compute(scores, errors).cpu()
        num_samples = error_rates.size(0)

        admissible_risks = (error_rates > self.risk_threshold) * 1
        max_cov_at_risk = admissible_risks.flip(0).argmin()

        # check if max_cov_at_risk is really admissible, if not return nan
        risk = admissible_risks[max_cov_at_risk]
        if risk > self.risk_threshold:
            out = torch.tensor([float("nan")])
        else:
            out = 1 - max_cov_at_risk / num_samples

        return dict(cov_at_risk=out.item())


@METRICS.register_module()
class CovAt5Risk(CovAtxRisk):
    """Coverage at 5% Risk.

    If there are multiple coverage values corresponding to 5% risk, the
    coverage at 5% risk is the maximum coverage value corresponding to 5%
    risk. If no there is no coverage value corresponding to the given risk,
    this metric returns float("nan").
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(risk_threshold=0.05, **kwargs)


@METRICS.register_module()
class RiskAtxCov(AURC):
    """Risk at given Coverage.

    Args:
        cov_threshold (float): The coverage threshold at which to compute
            the risk.
        kwargs: Additional arguments to pass to the metric class.
    """

    default_prefix: Optional[str] = "sc"

    def __init__(self, cov_threshold: float, **kwargs) -> None:
        super().__init__(**kwargs)
        _risk_coverage_checks(cov_threshold)
        self.cov_threshold = cov_threshold

    def compute_metrics(self, results: List) -> dict:
        scores = torch.cat([res["score"] for res in results])
        errors = torch.cat([res["error"] for res in results])

        error_rates = _aurc_rejection_rate_compute(scores, errors).cpu()
        error_rates = error_rates[math.ceil(scores.size(0) * self.cov_threshold) - 1]
        return dict(risk_at_cov=error_rates.item())


@METRICS.register_module()
class RiskAt80Cov(RiskAtxCov):
    """Risk at 80% Coverage."""

    def __init__(self, **kwargs) -> None:
        super().__init__(cov_threshold=0.8, **kwargs)
