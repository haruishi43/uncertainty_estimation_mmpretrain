# model settings
# minor modifications to the backbone
# based on the implementations related to EDL
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
