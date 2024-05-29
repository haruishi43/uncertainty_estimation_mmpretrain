_base_ = [
    "../_base_/datasets/cifar5_bs16.py",
    "../_base_/schedules/cifar10_short_bs128.py",
    "../_base_/default_runtime.py",
]

# dataset config
train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=128)

# model settings
model = dict(
    type="EvidentialImageClassifier",
    backbone=dict(
        type="ResNet_CIFAR", depth=18, num_stages=4, out_indices=(3,), style="pytorch"
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="EvidentialLinearClsHead",
        num_classes=5,
        in_channels=512,
        lamb=0.1,
        loss=dict(type="RelaxedDirichletSSELoss"),
    ),
)

custom_hooks = [
    dict(type="PassStepInfoHook"),
]
