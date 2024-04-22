_base_ = [
    "../_base_/datasets/mnist_bs128.py",
    # "../_base_/schedules/mnist_sgd_bs128.py",
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
        evidence_func="relu",
        loss=dict(type="DirichletMSELoss"),
    ),
)

custom_hooks = [
    dict(type="PassStepInfoHook"),
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.001)
)

# learning policy
param_scheduler = dict(
    type="MultiStepLR",  # learning policy, decay on several milestones.
    by_epoch=True,  # update based on epoch.
    milestones=[15],  # decay at the 15th epochs.
    gamma=0.1,  # decay to 0.1 times.
)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)
