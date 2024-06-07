_base_ = [
    "../_base_/models/lenet5.py",
    "../_base_/datasets/mnist_bs128.py",
    "../_base_/schedules/mnist_sgd_bs128.py",
    "../_base_/default_runtime.py",
]

# baseline (default) configuration
