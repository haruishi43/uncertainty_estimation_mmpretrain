_base_ = [
    "../_base_/models/resnet18_cifar.py",
    "../_base_/datasets/cifar5_bs128.py",
    "../_base_/schedules/cifar10_short_bs128.py",
    "../_base_/default_runtime.py",
]

# baseline (default) configuration
model = dict(head=dict(num_classes=5))
