_base_ = [
    "../_base_/models/lenet5.py",
    "../_base_/datasets/mnist_bs1000.py",
    "../_base_/schedules/mnist_bs1000.py",
    "../_base_/default_runtime.py",
]

# baseline (default) configuration

# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="ModernLeNet5",
        flattened=True,
        channels=[20, 50, 500],
        num_classes=10,
        act_cfg=dict(type="ReLU"),
    ),
    neck=None,
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
