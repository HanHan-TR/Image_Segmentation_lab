seed = 123
deterministic = True

epochs = 60

train_batch_size = 16
val_batch_size = 8

num_workers = 4

# optimizer
optimizer = dict(type='SGD',
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=0.0005)
# learning policy
lr_config = dict(type='StepLR',
                 step_size=15,
                 gamma=0.1)

# runtime settings
amp = True
