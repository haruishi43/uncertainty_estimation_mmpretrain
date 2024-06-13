#!/usr/bin/env python3

from typing import List, Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric

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
