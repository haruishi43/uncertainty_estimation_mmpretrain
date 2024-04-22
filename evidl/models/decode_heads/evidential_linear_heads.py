#!/usr/bin/env python3

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule, ModuleList

from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.registry import MODELS
from mmpretrain.models.heads.cls_head import ClsHead
from mmpretrain.structures import DataSample

from ..utils.layers import Dirichlet


class LinearBlock(BaseModule):
    """Linear block for StackedLinearClsHead."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_rate=0.0,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.fc = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        self.dropout = None

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        if act_cfg is not None:
            self.act = build_activation_layer(act_cfg)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """The forward process."""
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


@MODELS.register_module()
class EvidentialStackedLinearClsHead(ClsHead):
    """Stacked Linear Classifier head with Dirichlet output.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        mid_channels (Sequence[int]): Number of channels in the hidden fc
            layers.
        dropout_rate (float): Dropout rate after each hidden fc layer,
            except the last layer. Defaults to 0.
        norm_cfg (dict, optional): Config dict of normalization layer after
            each hidden fc layer, except the last layer. Defaults to None.
        act_cfg (dict, optional): Config dict of activation function after each
            hidden layer, except the last layer. Defaults to use "ReLU".
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        mid_channels: Sequence[int],
        dropout_rate: float = 0.0,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.in_channels = in_channels
        if self.num_classes <= 0:
            raise ValueError(f"num_classes={num_classes} must be a positive integer")

        assert isinstance(mid_channels, Sequence), (
            f"`mid_channels` of StackedLinearClsHead should be a sequence, "
            f"instead of {type(mid_channels)}"
        )
        self.mid_channels = mid_channels

        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self._init_layers()

    def _init_layers(self):
        """ "Init layers."""
        self.layers = ModuleList()
        in_channels = self.in_channels
        for hidden_channels in self.mid_channels:
            self.layers.append(
                LinearBlock(
                    in_channels,
                    hidden_channels,
                    dropout_rate=self.dropout_rate,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
            in_channels = hidden_channels

        self.dirichlet = Dirichlet(
            self.mid_channels[-1],
            self.num_classes,
        )

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage.
        """
        x = feats[-1]
        for layer in self.layers:
            if len(x.shape) == 4:
                x = x.squeeze()
            x = layer(x)
        return x

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)

        # We output the alpha values which is the "evidence + 1"
        alpha = self.dirichlet(pre_logits)
        return alpha

    def loss(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[DataSample],
        step: int,
        max_steps: int,
        **kwargs,
    ) -> dict:
        cls_score = self(feats)

        # Unpack data samples and pack targets
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(
            cls_score,
            target,
            avg_factor=cls_score.size(0),
            step=step,
            max_steps=max_steps,
            **kwargs,
        )
        losses["loss"] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, (
                "If you enable batch augmentation "
                "like mixup during training, `cal_acc` is pointless."
            )
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update({f"accuracy_top-{k}": a for k, a in zip(self.topk, acc)})

        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: Optional[List[Optional[DataSample]]] = None,
    ) -> List[DataSample]:
        alpha = self(feats)
        predictions = self._get_predictions(alpha, data_samples)
        return predictions

    def _get_predictions(self, alpha, data_samples):

        # this is actually a normalized probability map
        # similar to softmax
        alpha = alpha.detach()
        pred_scores = alpha / alpha.sum(dim=1, keepdim=True)

        # do something with uncertainty
        pred_uncerts = self.num_classes / alpha.sum(dim=1, keepdim=True)

        # predicted labels
        pred_labels = alpha.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, uncert, label in zip(
            data_samples,
            pred_scores,
            pred_uncerts,
            pred_labels,
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
