# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="ModernLeNet5",
        num_classes=10,
        act_cfg=dict(type="ReLU"),
    ),
    neck=None,
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
