_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/datasets/h3wb_double_hands3d.py'
]


evaluation = dict(
    interval=10, 
    metric=['mpjpe', 'p-mpjpe'], 
    save_best='MPJPE')


# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    by_epoch=False,
    step=100000,
    gamma=0.96,
)


total_epochs = 200

channel_cfg = dict(
    num_input_channels=48,
    num_output_channels= 6 * 42,
    dataset_joints=48,
    dataset_channel=[
       list(range(48)),
    ],
    inference_channel=list(range(48)))


# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * channel_cfg['num_input_channels'],
        stem_channels=1024,
        num_blocks=4,
        kernel_sizes=(1, 1, 1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='HandsTemporalRegressionHead',
        in_channels=1024,
        num_joints= 6 * 42,
        num_root_joints=6,
        loss_keypoint=dict(type='MSELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))  

# data settings
data_root = 'data/h36m'
data_cfg = dict(
    num_joints=48,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
)


train_pipeline = [
    dict(
        type='CustomGetRootCenteredPose',
        pre_root_num=6),
    dict(
        type='NormalizeJointCoordinate',
        item='target',
        norm_param_file=f'{data_root}/annotation_wholebody3d/double_hands_joint3d_rel_stats.pkl'),
    dict(
        type='NormalizeJointCoordinate',
        item='input_2d',
        norm_param_file=f'{data_root}/annotation_wholebody3d/double_hands_joint2d_stats.pkl'),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'target_mean', 'target_std'
        ])
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=512),
    test_dataloader=dict(samples_per_gpu=512),
    train=dict(
        type='DoubleHands3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/h3wb_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='DoubleHands3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/h3wb_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='DoubleHands3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/h3wb_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)