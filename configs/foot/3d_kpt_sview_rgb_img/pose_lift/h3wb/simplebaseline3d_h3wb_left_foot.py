_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h3wb_foot3d.py'
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


# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 4,
        stem_channels=1024,
        num_blocks=2,
        kernel_sizes=(1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=4-1,  # do not predict root joint
        loss_keypoint=dict(type='MSELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/h3wb'
data_cfg = dict(
    num_joints=4,
    seq_len=1,
    seq_frame_interval=1,
    causal=True,
    joint_2d_src='gt',
    need_camera_param=False,
    camera_param_file=f'{data_root}/annotations/cameras.pkl',
)

# 3D joint normalization parameters
# From file: '{data_root}/annotations/joint3d_rel_stats.pkl'
joint_3d_normalize_param = dict(
    mean=[[-0.00100205,  0.05328414,  0.01158549],
       [-0.00083378,  0.0559862 ,  0.01225291],
       [-0.00045601,  0.04030471,  0.00915677]],
    std=[[0.09650524, 0.04471978, 0.14004999],
       [0.08712755, 0.03457024, 0.10180451],
       [0.02762162, 0.01381713, 0.03833972]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[532.90338803, 576.78355853],
       [533.56692915, 588.95513987],
       [533.65039025, 589.21284974],
       [532.47219793, 585.468885  ]],
    std=[[102.99067186,  50.39087227],
       [103.53227537,  51.7706836 ],
       [102.80298566,  51.19201958],
       [103.84026609,  51.53965411]])

train_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=True),
    dict(
        type='NormalizeJointCoordinate',
        item='target',
        mean=joint_3d_normalize_param['mean'],
        std=joint_3d_normalize_param['std']),
    dict(
        type='NormalizeJointCoordinate',
        item='input_2d',
        mean=joint_2d_normalize_param['mean'],
        std=joint_2d_normalize_param['std']),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'root_position_index', 'target_mean', 'target_std'
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
        type='Foot3DH3WBDataset',
        ann_file=f'{data_root}/annotations/train_left_foot.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Foot3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_left_foot.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Foot3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_left_root.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
