_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/datasets/h3wb_lefthand3d.py'
]

evaluation = dict(
    interval=10, 
    metric=['mpjpe', 'p-mpjpe'], 
    save_best='MPJPE')

optimizer = dict(
    type='Adam',
    lr=1e-3)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    by_epoch=False,
    step=2000,
    gamma=0.96)

total_epochs = 200

channel_cfg = dict(
    num_input_channels=22,
    num_output_channels= 2 * 21,
    dataset_joints=22,
    dataset_channel=[
       list(range(22))],
    inference_channel=list(range(22)))

model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * channel_cfg['num_input_channels'],
        stem_channels=1024,
        num_blocks=3,
        kernel_sizes=(1, 1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='HandLocalAndGlobalTemporalRegressionHead',
        in_channels=1024,
        num_joints= 2 * 21,
        real_num_joints=21,
        loss_keypoint=dict(type='MSELoss'),
        loss_rel_keypoint=dict(type='SmoothL1Loss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))  

data_root = 'data/h36m'
data_cfg = dict(
    num_joints=22,
    seq_len=1,
    seq_frame_interval=1,
    causal=True)

train_pipeline = [
    dict(
        type='CustomGetLocalAndGlobalOffsets',
        parents=[-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 1, 10, 11, 12, 1, 14, 15, 16, 1, 18, 19, 20]),
    dict(
        type='NormalizeJointCoordinate',
        item='target',
        norm_param_file=f'{data_root}/annotation_wholebody3d/lhand_joint3d_local_global_stats.pkl'),
    dict(
        type='NormalizeJointCoordinate',
        item='input_2d',
        norm_param_file=f'{data_root}/annotation_wholebody3d/lhand_joint2d_stats.pkl'),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'target_mean', 'target_std', 'parents'])
    ]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=512),
    test_dataloader=dict(samples_per_gpu=512),
    train=dict(
        type='LocalAndGlobalHand3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/lhand_h3wb_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='LocalAndGlobalHand3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/lhand_h3wb_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='LocalAndGlobalHand3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/lhand_h3wb_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

