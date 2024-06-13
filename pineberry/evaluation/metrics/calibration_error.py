#!/usr/bin/env python3

"""Expected Calibration Error (ECE) metric."""

from typing import Any, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from mmengine.evaluator import BaseMetric

from mmpretrain.evaluation.metrics.single_label import to_tensor
from mmpretrain.registry import METRICS


def _binning_bucketize(
    confidences: Tensor, accuracies: Tensor, bin_boundaries: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities

    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = torch.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    conf_bin = torch.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    count_bin = torch.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )

    indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)

    prop_bin = count_bin / count_bin.sum()
    return acc_bin, conf_bin, prop_bin


@METRICS.register_module()
class CalibrationError(BaseMetric):

    default_prefix: Optional[str] = "calib_err"

    def __init__(
        self,
        n_bins: int = 15,
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
                result["confidence"] = pred
                result["accuracy"] = target
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

        ece = self.calculate(
            confidences=confidences,
            accuracies=accuracies,
            bin_boundaries=self.n_bins,
            norm=self.norm,
        )
        ece = ece.item()
        return dict(ECE=ece)

    @staticmethod
    def calculate(
        confidences: Union[Tensor, np.ndarray, Sequence],
        accuracies: Union[Tensor, np.ndarray, Sequence],
        bin_boundaries: Union[Tensor, int],
        norm: str = "l1",
        debias: bool = False,
    ) -> Tensor:
        """Compute the calibration error given the provided bin boundaries and norm.

        Modified from `torchmetrics.classification.calibration_error`.

        Args:
            confidences: The confidence (i.e. predicted prob) of the top1 prediction.
            accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
            bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
            norm: Norm function to use when computing calibration error. Defaults to "l1".
            debias: Apply debiasing to L2 norm computation as in
                `Verified Uncertainty Calibration`_. Defaults to False.

        Raises:
            ValueError: If an unsupported norm function is provided.

        Returns:
            Tensor: Calibration error scalar.

        """

        confidences = to_tensor(confidences).float()
        accuracies = to_tensor(accuracies).float()

        if isinstance(bin_boundaries, int):
            bin_boundaries = torch.linspace(
                0,
                1,
                bin_boundaries + 1,
                dtype=confidences.dtype,
                device=confidences.device,
            )

        if norm not in {"l1", "l2", "max"}:
            raise ValueError(
                f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}"
            )

        with torch.no_grad():
            acc_bin, conf_bin, prop_bin = _binning_bucketize(
                confidences, accuracies, bin_boundaries
            )

        if norm == "l1":
            return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
        if norm == "max":
            ce = torch.max(torch.abs(acc_bin - conf_bin))
        if norm == "l2":
            ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
            # NOTE: debiasing is disabled in the wrapper functions. This implementation differs from that in sklearn.
            if debias:
                # the order here (acc_bin - 1 ) vs (1 - acc_bin) is flipped from
                # the equation in Verified Uncertainty Prediction (Kumar et al 2019)/
                debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                    prop_bin * accuracies.size()[0] - 1
                )
                ce += torch.sum(
                    torch.nan_to_num(debias_bins)
                )  # replace nans with zeros if nothing appeared in a bin
            return torch.sqrt(ce) if ce > 0 else torch.tensor(0)
        return ce
