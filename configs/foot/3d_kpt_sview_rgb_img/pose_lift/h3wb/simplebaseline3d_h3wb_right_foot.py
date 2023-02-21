_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h3wb_right_foot.py'
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
data_root = 'data/h36m'
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
# joint_3d_normalize_param = dict(
#     mean=[[-0.00107147,  0.05438742,  0.0120387 ],
#        [-0.00102378,  0.05488608,  0.01226583],
#        [-0.00044053,  0.04144828,  0.00939285]],
#     std=[[0.09824096, 0.04529132, 0.13779035],
#        [0.08629216, 0.03634149, 0.11017919],
#        [0.02706109, 0.01436437, 0.04027776]])

joint_3d_normalize_param = dict(
    mean=[[-0.00161736,  0.05592401,  0.01225325],
       [-0.00148055,  0.05603045,  0.01256697],
       [-0.00032021,  0.04130705,  0.00933195]],
    std=[[0.09636132, 0.04579895, 0.13401479],
       [0.08414907, 0.03684599, 0.10698585],
       [0.02659306, 0.01392847, 0.03911938]])


# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
# joint_2d_normalize_param = dict(
#     mean=[[531.54909992, 576.87212007],
#        [531.52146222, 589.35963141],
#        [531.06593207, 589.46362616],
#        [531.3871354 , 585.76267864]],
#     std=[[111.80010231,  52.13738756],
#        [112.69958285,  53.80829198],
#        [116.15619428,  54.00611491],
#        [111.78637547,  53.28889042]])

joint_2d_normalize_param = dict(
    mean=[[529.49252607, 577.10985614],
       [529.36160546, 590.17255573],
       [528.79651303, 590.15445762],
       [529.36946405, 585.92209152]],
    std=[[119.23282497,  52.24963078],
       [119.60772276,  55.17550538],
       [123.25922294,  55.21895105],
       [119.43246198,  53.16351984]])


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
        ann_file=f'{data_root}/annotation_wholebody3d/rfoot_h3wb_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Foot3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/rfoot_h3wb_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Foot3DH3WBDataset',
        ann_file=f'{data_root}/annotation_wholebody3d/rfoot_h3wb_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
