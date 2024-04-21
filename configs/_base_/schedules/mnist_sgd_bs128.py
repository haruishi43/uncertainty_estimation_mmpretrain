# optimizer
optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
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
