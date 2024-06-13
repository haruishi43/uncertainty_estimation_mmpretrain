#!/usr/bin/env python3

"""Base Classification Head for EDL."""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample


@MODELS.register_module()
class EDLClsHead(BaseModule):
    """EDL Classification head.

    Args:
        loss (dict, list): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss: Union[dict, list] = [
            dict(type="EDLSSELoss", loss_weight=1.0),
            dict(type="EDLKLDivLoss", loss_weight=1.0),
        ],
        cal_acc: bool = False,
        topk: Union[int, Tuple[int]] = (1,),
        cal_uncert: bool = False,
        edl_layer: dict = dict(
            type="DirichletLayer",
            evidence_function="softplus",
        ),
        lamb: float = 1.0,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        init_cfg: Optional[dict] = dict(type="Normal", layer="Linear", std=0.01),
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_classes = num_classes
        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        self.topk = topk
        self.cal_acc = cal_acc

        # build loss module
        if isinstance(loss, dict):
            loss = MODELS.build(loss)
        if isinstance(loss, (list, tuple)):
            loss = nn.ModuleList([MODELS.build(l) for l in loss])
        self.loss_module = loss

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.edl_layer_cfg = edl_layer
        self.lamb = lamb

        self._init_layers()

    def _init_layers(self):
        """Init layers."""
        edl_layer = deepcopy(self.edl_layer_cfg)
        edl_layer["in_channels"] = self.in_channels
        edl_layer["out_channels"] = self.num_classes
        self.edl_layer = MODELS.build(edl_layer)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``ClsHead``, we just obtain the feature
        of the last stage.
        """
        # The ClsHead doesn't have other module, just return after unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)

        # EDL layer
        evidence = self.edl_layer(pre_logits)
        return evidence

    def loss(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[DataSample],
        **kwargs,
    ) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        evidence = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(evidence, data_samples, **kwargs)
        return losses

    def _get_loss(
        self,
        evidence: torch.Tensor,
        data_samples: List[DataSample],
        **kwargs,
    ) -> dict:
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        losses = dict()
        if not isinstance(self.loss_module, nn.ModuleList):
            loss_modules = [self.loss_module]
        else:
            loss_modules = self.loss_module
        for loss_module in loss_modules:
            loss_name = loss_module.loss_name
            loss = loss_module(
                evidence,
                target,
                avg_factor=evidence.size(0),
                **kwargs,
            )
            if loss_name not in losses:
                losses[loss_name] = loss
            else:
                losses[loss_name] += loss

        alpha = evidence + self.lamb
        cls_score = alpha / alpha.sum(dim=1, keepdim=True)

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, (
                "If you enable batch augmentation "
                "like mixup during training, `cal_acc` is pointless."
            )
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update({f"accuracy_top-{k}": a for k, a in zip(self.topk, acc)})

        # TODO: add uncertainty related metrics

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None,
    ) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample | None], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        evidence = self(feats)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(evidence, data_samples)
        return predictions

    def _get_predictions(
        self,
        evidence: torch.Tensor,
        data_samples: Optional[List[Optional[DataSample]]] = None,
    ) -> List[DataSample]:
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """

        evidence = evidence.detach()
        alpha = evidence + self.lamb
        S = alpha.sum(dim=1, keepdim=True)
        pred_scores = alpha / S

        # epistemic uncertainty
        pred_uncerts = self.lamb * self.num_classes / S

        # predicted labels
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, uncert, label in zip(
            data_samples, pred_scores, pred_uncerts, pred_labels
        ):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)

            # NOTE add uncertainty as a custom filed
            data_sample.set_field(
                uncert,
                "uncertainty",
                dtype=torch.Tensor,
            )

            out_data_samples.append(data_sample)
        return out_data_samples
