from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmpretrain.evaluation.metrics.single_label import to_tensor
from mmpretrain.registry import METRICS


@METRICS.register_module()
class PredictiveDistributionECDF(BaseMetric):
    """Measures ECDF of the Predictive Distribution.

    The CDF function is binned by the entropy (x-axis) and the values are
    the maximum probability of the predictive distribution (y-axis).

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = "ecdf_pred_dist"

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()

            assert "pred_score" in data_sample, "The prediction score must be provided."
            result["pred_score"] = data_sample["pred_score"].cpu()

            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # concat
        pred = torch.stack([res["pred_score"] for res in results])

        # Add a small constant to avoid log(0)
        print(pred.shape)
        entropies = -(pred * torch.log(pred + 1e-9)).sum(dim=1)

        # calculate empirical cdf
        sorted_entropies, _ = torch.sort(entropies)
        cdf = torch.arange(1, len(sorted_entropies) + 1) / len(sorted_entropies)

        # for plot
        metrics["metric"] = [
            sorted_entropies.numpy(),
            cdf.numpy(),
        ]

        return metrics


@METRICS.register_module()
class UncertaintyThresholdedAccuracy(BaseMetric):
    """Measures the accuracies by thresholding by uncertainties.

    For either binary classification or multi-class classification, the
    accuracy is the fraction of correct predictions in all predictions:

    Args:
        uncert_bins (int): The number of bins to divide the uncertainty.
            Defaults to 20.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = "uncert_thresh_acc"

    def __init__(
        self,
        uncert_bins: int = 20,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        assert uncert_bins > 0, "The number of bins must be greater than 0."
        self.uncert_bins = uncert_bins

    def process(self, data_batch, data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            result = dict()
            assert "pred_score" in data_sample, "The prediction score must be provided."
            result["pred_score"] = data_sample["pred_score"].cpu()

            assert "uncertainty" in data_sample, "The uncertainty must be provided."
            result["uncertainty"] = data_sample["uncertainty"].cpu()

            result["gt_label"] = data_sample["gt_label"].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # concat
        target = torch.cat([res["gt_label"] for res in results])
        pred = torch.stack([res["pred_score"] for res in results])
        uncert = torch.cat([res["uncertainty"] for res in results])

        try:
            acc = self.calculate(pred, target, uncert, self.uncert_bins)
            # should be a list of tensor
        except ValueError as e:
            raise ValueError(
                str(e) + " Please check the `val_evaluator` and "
                "`test_evaluator` fields in your config file."
            )

        metrics["metric"] = [a.item() for a in acc]

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        uncert: Union[torch.Tensor, np.ndarray, Sequence],
        uncert_bins: int = 20,
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """Calculate the accuracy.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            uncert_bins (int): The number of bins to divide the uncertainty.
                Defaults to 20.

        Returns:
            List[torch.Tensor]: Accuracy.
        """

        pred = to_tensor(pred).float()
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), (
            f"The size of pred ({pred.size(0)}) doesn't match "
            f"the target ({target.size(0)})."
        )

        assert pred.ndim > 1, "The prediction must be scores."

        _, pred_label = pred.max(dim=1)
        pred_label = pred_label.t()
        correct = pred_label.eq(target)
        results = []
        for b in range(uncert_bins + 1):
            uncert_lb = b / uncert_bins

            # reject samples with uncertainty larger than the threshold
            uncert_mask = uncert <= uncert_lb
            _correct = correct & uncert_mask
            total = uncert_mask.float().sum(0, keepdim=True)

            if total == 0:
                acc = torch.tensor(100.0)
            else:
                _correct = _correct.float().sum(0, keepdim=True)
                acc = _correct.mul_(100.0 / total)

            results.append(acc)

        return results
