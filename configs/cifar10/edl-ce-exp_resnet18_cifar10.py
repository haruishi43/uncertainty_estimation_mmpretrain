_base_ = [
    "../_base_/datasets/cifar10_bs128.py",
    "../_base_/schedules/cifar10_short_bs128.py",
    "../_base_/default_runtime.py",
]

# model settings
model = dict(
    type="EvidentialImageClassifier",
    backbone=dict(
        type="ResNet_CIFAR",
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="EDLClsHead",
        num_classes=10,
        in_channels=512,
        loss=[
            dict(type="EDLCELoss", loss_weight=1.0),
            dict(type="EDLKLDivLoss", loss_weight=1.0),
        ],
        edl_layer=dict(
            type="DirichletLayer",
            evidence_function="exp",
        ),
    ),
)

val_evaluator = [
    dict(type="Accuracy"),
    dict(type="CalibrationError"),
    dict(type="AdaptiveCalibrationError"),
    dict(type="BrierScore"),
    dict(type="CategoricalNLL"),
    dict(type="Entropy"),
    dict(type="AURC"),
    dict(type="CovAt5Risk"),
    dict(type="RiskAt80Cov"),
]
test_evaluator = val_evaluator
