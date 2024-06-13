#!/usr/bin/env python3

"""Adaptive Expected Calibration Error (aECE) metric."""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from mmengine.evaluator import BaseMetric

from mmpretrain.evaluation.metrics.single_label import to_tensor
from mmpretrain.registry import METRICS


def _equal_binning_bucketize(
    confidences: Tensor, accuracies: Tensor, num_bins: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute bins for the adaptive calibration error.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1
            prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration
            error.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities
    """
    confidences, indices = torch.sort(confidences)
    accuracies = accuracies[indices]
    acc_bin, conf_bin = (
        accuracies.tensor_split(num_bins),
        confidences.tensor_split(num_bins),
    )
    count_bin = torch.as_tensor(
        [len(cb) for cb in conf_bin],
        dtype=confidences.dtype,
        device=confidences.device,
    )
    return (
        pad_sequence(acc_bin, batch_first=True).sum(1) / count_bin,
        pad_sequence(conf_bin, batch_first=True).sum(1) / count_bin,
        torch.as_tensor(count_bin) / len(confidences),
    )


@METRICS.register_module()
class AdaptiveCalibrationError(BaseMetric):

    default_prefix: Optional[str] = "calib_err"

    def __init__(
        self,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l2",
        binary: bool = False,
    ) -> None:
        super().__init__()

        self.n_bins = n_bins
        self.norm = norm
        self.binary = binary

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = dict()

            # not one-hot
            if "gt_score" in data_sample:
                target = data_sample["gt_score"].clone()
                if target.ndim == pred.ndim:
                    target = target.argmax(dim=1)
            else:
                target = data_sample["gt_label"].clone()

            if self.binary:
                pred = data_sample["pred_score"].clone()

                conf, label = torch.max(pred, 1 - pred), torch.round(pred)
                acc = label.eq(target).float()
                result["confidence"] = conf
                result["accuracy"] = acc
            else:
                # TODO: properly handle uncertainty for ECE
                # if 'uncertainty' in data_sample:
                #     # HACK: is there a cleaner way of doing this?
                #     # explicit model confidence (e.g. edl)
                #     uncert = data_sample['uncertainty'].clone()
                #     conf = (1 - uncert).clip(0, 1)
                #     label = data_sample["pred_label"].clone()
                # else:
                #     pred = data_sample["pred_score"].clone()
                #     conf, label = pred.max(dim=-1)
                pred = data_sample["pred_score"].clone()
                conf, label = pred.max(dim=-1)
                result["confidence"] = conf.squeeze()
                result["accuracy"] = label.eq(target).squeeze()

            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        confidences = torch.stack([res["confidence"] for res in results])
        accuracies = torch.stack([res["accuracy"] for res in results])

        aece = self.calculate(
            confidences=confidences,
            accuracies=accuracies,
            num_bins=self.n_bins,
            norm=self.norm,
        )
        aece = aece.item()
        return dict(aECE=aece)

    @staticmethod
    def calculate(
        confidences: Union[Tensor, np.ndarray, Sequence],
        accuracies: Union[Tensor, np.ndarray, Sequence],
        num_bins: int,
        norm: Literal["l1", "l2", "max"] = "l1",
        debias: bool = False,
    ) -> Tensor:
        """Compute the adaptive calibration error given the provided number of bins
            and norm.

        Args:
            confidences: The confidence (i.e. predicted prob) of the top1
                prediction.
            accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
            num_bins: Number of bins to use when computing adaptive calibration
                error.
            norm: Norm function to use when computing calibration error. Defaults
                to "l1".
            debias: Apply debiasing to L2 norm computation as in
                `Verified Uncertainty Calibration`_. Defaults to False.

        Returns:
            Tensor: Adaptive Calibration error scalar.
        """

        confidences = to_tensor(confidences).float()
        accuracies = to_tensor(accuracies).float()

        with torch.no_grad():
            acc_bin, conf_bin, prop_bin = _equal_binning_bucketize(
                confidences, accuracies, num_bins
            )

        if norm == "l1":
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if norm == "max":
            ace = torch.max(torch.abs(acc_bin - conf_bin))
        if norm == "l2":
            ace = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
            if debias:  # coverage: ignore
                debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                    prop_bin * accuracies.size()[0] - 1
                )
                ace += torch.sum(
                    torch.nan_to_num(debias_bins)
                )  # replace nans with zeros if nothing appeared in a bin
            return torch.sqrt(ace) if ace > 0 else torch.tensor(0)
        return ace
