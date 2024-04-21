#!/usr/bin/env python3

import torch.nn as nn

from mmcv.cnn import ConvModule, build_activation_layer

from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.base_backbone import BaseBackbone


@MODELS.register_module()
class ModernLeNet5(BaseBackbone):
    """`LeNet5 <https://en.wikipedia.org/wiki/LeNet>`_ backbone.

    The input for LeNet-5 is a 32×32 grayscale image.

    Args:
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.

    This version is different from the original LeNet5 in that it uses
    ReLU activation functions instead of sigmoid and tanh.
    """

    def __init__(
        self,
        num_classes: int = -1,
        act_cfg: dict = dict(type="ReLU"),
        conv_cfg: dict = dict(type="Conv2d"),
        norm_cfg: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.features = nn.Sequential(
            ConvModule(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                act_cfg=act_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
            nn.AvgPool2d(kernel_size=2),
            ConvModule(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                act_cfg=act_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
            nn.AvgPool2d(kernel_size=2),
            ConvModule(
                in_channels=16,
                out_channels=120,
                kernel_size=5,
                stride=1,
                act_cfg=act_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            ),
        )
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(120, 84),
                build_activation_layer(act_cfg),
                nn.Linear(84, num_classes),
            )

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x.squeeze())

        return (x,)
