_base_ = [
    "../_base_/models/resnet18_cifar.py",
    "../_base_/datasets/cifar10_bs128.py",
    "../_base_/schedules/cifar10_short_bs128.py",
    "../_base_/default_runtime.py",
]

val_evaluator = [
    dict(type="Accuracy"),
    dict(type="CategoricalNLL"),
    dict(type="CalibrationError"),
    dict(type="AdaptiveCalibrationError"),
    dict(type="BrierScore"),
    dict(type="Entropy"),
    dict(type="AURC"),
    dict(type="CovAt5Risk"),
    dict(type="RiskAt80Cov"),
]
test_evaluator = val_evaluator
