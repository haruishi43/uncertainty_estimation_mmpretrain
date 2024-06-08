_base_ = [
    "../_base_/datasets/mnist_bs1000.py",
    "../_base_/schedules/mnist_bs1000.py",
    "../_base_/default_runtime.py",
]

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
        type="EDLClsHead",
        num_classes=10,
        in_channels=500,
        loss=[
            dict(type="EDLCELoss", loss_weight=1.0),
            dict(type="EDLKLDivLoss", loss_weight=1.0),
        ],
        edl_layer=dict(
            type="DirichletLayer",
            evidence_function="exp_tanh",
        ),
    ),
)
