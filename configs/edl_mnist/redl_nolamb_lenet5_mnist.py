_base_ = [
    "../_base_/datasets/mnist_bs128.py",
    "../_base_/schedules/mnist_sgd_bs128.py",
    "../_base_/default_runtime.py",
]

# baseline (default) configuration

model = dict(
    type="EvidentialImageClassifier",
    backbone=dict(
        type="ModernLeNet5",
        act_cfg=dict(type="ReLU"),
    ),
    neck=None,
    head=dict(
        type="EvidentialStackedLinearClsHead",
        num_classes=10,
        in_channels=120,
        mid_channels=[84],
        lamb=1.0,
        loss=dict(type="RelaxedEDLSSELoss"),
    ),
)
