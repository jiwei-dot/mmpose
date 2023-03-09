_base_ = [
    '../_base_/default_runtime.py',
]

evaluation = dict(
    interval=10, 
    metric=['precision', 'recall', 'f1_score'],
    save_best='f1_score',
    greater_keys=['f1_score', ]
)

optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    by_epoch=False,
    step=10000,
    gamma=0.96,
)


total_epochs = 100

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])


# model setting
model = dict(
    type='FootContactor',
    pretrained=None,
    backbone = dict(
        type='TCN',
        in_channels=13 * 3,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(3, 3, 1),
        dropout=0.25,
        use_stride_conv=True),
    head=dict(
        type='CustomTemporalClassificationHead',
        in_channels=1024),
    train_cfg=dict(),
    test_cfg=dict()
)


# data setting
data_root = 'data/foot_contact'
data_cfg = dict(
    seq_len=9,
    causal=False
)


train_pipeline = [
    dict(
        type='CustomNormalizeCoordinates',
        pixel_length=200.45647751632757,
    ),
    dict(
        type='CustomRelativeToCenterRoot',
        root_index=8,
    ),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['imgname', ]
    )
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='FootContactDataset',
        ann_file=f'{data_root}/annotations/foot_contact_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='FootContactDataset',
        ann_file=f'{data_root}/annotations/foot_contact_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='FootContactDataset',
        ann_file=f'{data_root}/annotations/foot_contact_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline)
)