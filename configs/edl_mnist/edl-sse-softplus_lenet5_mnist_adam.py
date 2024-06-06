_base_ = [
    # "../_base_/datasets/mnist_bs128.py",
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
        evidence_func="softplus",
        loss=dict(type="EDLSSELoss"),
    ),
)

# randomness = dict(seed=0)
# optimizer
optim_wrapper = dict(
    optimizer=dict(type="Adam", lr=0.001),
)

# learning policy
param_scheduler = dict(
    type="MultiStepLR",  # learning policy, decay on several milestones.
    by_epoch=True,  # update based on epoch.
    milestones=[7],  # decay at the 7th epochs.
    gamma=0.1,  # decay to 0.1 times.
)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# dataset settings
dataset_type = "MNIST"

# NOTE: the Runner automatically merges `data_preprocessor` to `model`
data_preprocessor = dict(mean=[33.46], std=[78.87], num_classes=10)

pipeline = [
    dict(type="Resize", scale=32),
    dict(type="PackInputs"),
]

common_data_cfg = dict(
    type=dataset_type,
    data_prefix="data/mnist",
    pipeline=pipeline,
)

train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(**common_data_cfg, test_mode=False),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=1000,
    num_workers=2,
    dataset=dict(**common_data_cfg, test_mode=True),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="Accuracy", topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator


# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=128)
