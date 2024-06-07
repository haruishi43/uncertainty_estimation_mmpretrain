_base_ = [
    "../_base_/datasets/mnist_bs1000.py",
    "../_base_/schedules/mnist_adam.py",
    "../_base_/default_runtime.py",
]

# baseline (default) configuration

model = dict(
    type="EvidentialImageClassifier",
    backbone=dict(
        type="ModernLeNet5",
        flattened=True,
        channels=[20, 50, 500],
        act_cfg=dict(type="ReLU"),
    ),
    neck=None,
    head=dict(
        type="EvidentialLinearClsHead",
        num_classes=10,
        in_channels=500,
        evidence_func="exp",
        loss=dict(type="EDLSSELoss"),
    ),
)
