#!/usr/bin/env python3

from typing import List

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
        channels: List[int] = [6, 16, 120],
        mid_channels: int = 84,
        flattened: bool = False,
        act_cfg: dict = dict(type="ReLU"),
        conv_cfg: dict = dict(type="Conv2d"),
        norm_cfg: dict = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.flattened = flattened

        assert len(channels) == 3

        if self.flattened:
            self.features = nn.Sequential(
                nn.Conv2d(1, channels[0], 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(channels[0], channels[1], 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(channels[1] * 5 * 5, channels[2]),
                nn.ReLU(),
            )
        else:
            # Difference:
            # last layer is conv (no flatten)
            # avgpool instead of maxpooling
            self.features = nn.Sequential(
                ConvModule(
                    in_channels=1,
                    out_channels=channels[0],
                    kernel_size=5,
                    stride=1,
                    # padding=2,
                    act_cfg=act_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ),
                nn.AvgPool2d(kernel_size=2),
                ConvModule(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=5,
                    stride=1,
                    act_cfg=act_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ),
                nn.AvgPool2d(kernel_size=2),
                ConvModule(
                    in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=5,
                    stride=1,
                    act_cfg=act_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                ),
            )

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                build_activation_layer(act_cfg),
                nn.Dropout(0.5),
                nn.Linear(channels[2], num_classes),
            )

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            x = self.classifier(x.squeeze())

        return (x,)
