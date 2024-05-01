#!/usr/bin/env python3

from typing import List

from mmpretrain.registry import DATASETS
from mmpretrain.datasets import CIFAR10


CIFAR5_CATEGORIES = ("airplane", "automobile", "bird", "cat", "deer")
LAST_5_CATEGORIES = ("dog", "frog", "horse", "ship", "truck")


@DATASETS.register_module()
class CIFAR5(CIFAR10):
    """CIFAR5 dataset for uncertainty estimation.

    Train the model using the first 5 classes.
    The later 5 classes are used for testing uncertainty estimation.
    """

    METAINFO = {"classes": CIFAR5_CATEGORIES}

    def __init__(
        self,
        use_unseen_classes: bool = False,
        **kwargs,
    ) -> None:
        self.use_unseen_classes = use_unseen_classes
        if use_unseen_classes:
            self.METAINFO = {"classes": LAST_5_CATEGORIES}
        super().__init__(**kwargs)

    def filter_data(self) -> List[dict]:
        """This is called from `full_init()`"""

        # usually, I would make use of `self.filter_cfg`, but
        # for simplicity, I will just filter it deterministically

        data_list = []
        for data in self.data_list:
            if self.use_unseen_classes:
                if data["gt_label"] >= 5:
                    data_list.append(data)
            else:
                if data["gt_label"] < 5:
                    data_list.append(data)

        return data_list
