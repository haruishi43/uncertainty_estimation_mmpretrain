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
