_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/datasets/h3wb_hand3d.py'
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
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[
       list(range(21)),
    ],
    inference_channel=list(range(21)))

# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 21,
        stem_channels=1024,
        num_blocks=3,
        kernel_sizes=(1, 1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=21-1,  # do not predict root joint (both left and right)
        loss_keypoint=dict(type='MSELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/h3wb'
data_cfg = dict(
    num_joints=21,
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
    mean=[[-0.00011647,  0.00526356,  0.00118197],
       [-0.00021845,  0.00889558,  0.00199454],
       [-0.00034686,  0.0163759 ,  0.00365739],
       [-0.00050133,  0.02706152,  0.00604585],
       [-0.00030557,  0.01467714,  0.00323015],
       [-0.00048893,  0.02696545,  0.00597458],
       [-0.00061396,  0.03574225,  0.00794873],
       [-0.00066918,  0.03975478,  0.00885624],
       [-0.00035337,  0.02060331,  0.00454214],
       [-0.00056001,  0.03444292,  0.0076424 ],
       [-0.00066024,  0.04169997,  0.00927936],
       [-0.00070646,  0.04537973,  0.01011497],
       [-0.00041634,  0.02746659,  0.00607971],
       [-0.00060224,  0.04052402,  0.00900499],
       [-0.000683  ,  0.0461355 ,  0.01027443],
       [-0.00073267,  0.04994837,  0.01113973],
       [-0.00047536,  0.03461963,  0.00768744],
       [-0.0006013 ,  0.04299816,  0.00956331],
       [-0.00064995,  0.04643822,  0.01034207],
       [-0.00070843,  0.05019757,  0.01119477]],
    std=[[0.01919336, 0.01815753, 0.02023605],
       [0.03811821, 0.03646607, 0.04013504],
       [0.04998096, 0.05114657, 0.05378379],
       [0.06030745, 0.06355943, 0.0644862 ],
       [0.04505542, 0.05105076, 0.05143163],
       [0.05757338, 0.06536192, 0.0648722 ],
       [0.06493638, 0.07192831, 0.07236102],
       [0.07095689, 0.07602542, 0.07819779],
       [0.04248138, 0.04889345, 0.04847557],
       [0.0554206 , 0.06289887, 0.06256232],
       [0.06216861, 0.06812862, 0.06943101],
       [0.0673411 , 0.07235795, 0.07497039],
       [0.04002912, 0.04559448, 0.04493514],
       [0.05061483, 0.05801898, 0.05701267],
       [0.05670394, 0.06217048, 0.06388143],
       [0.06182107, 0.06580128, 0.06922457],
       [0.03813567, 0.04085478, 0.04162662],
       [0.04586551, 0.04981376, 0.05111203],
       [0.04960132, 0.05345507, 0.05566515],
       [0.05358445, 0.05666155, 0.0606165 ]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[537.17816, 378.6269 ],
       [537.115  , 379.81943],
       [537.05884, 380.62274],
       [537.0357 , 382.34378],
       [536.9677 , 384.82288],
       [537.2089 , 381.97702],
       [537.15234, 384.8277 ],
       [537.0622 , 386.8681 ],
       [536.98804, 387.7882 ],
       [537.244  , 383.38898],
       [537.1385 , 386.597  ],
       [537.04193, 388.2808 ],
       [536.98346, 389.12186],
       [537.2154 , 385.01666],
       [537.1292 , 388.04437],
       [537.04596, 389.3379 ],
       [536.96906, 390.21625],
       [537.17126, 386.70947],
       [537.11926, 388.6537 ],
       [537.06384, 389.44052],
       [536.99176, 390.30853]],
    std=[[102.240715,  68.92654 ],
       [102.45372 ,  71.03538 ],
       [102.76883 ,  73.317375],
       [103.51732 ,  75.47354 ],
       [104.20534 ,  77.48764 ],
       [104.36836 ,  75.55355 ],
       [105.164444,  77.83126 ],
       [105.50196 ,  78.94471 ],
       [105.76016 ,  79.67556 ],
       [104.79712 ,  75.40221 ],
       [105.514435,  77.66023 ],
       [105.7492  ,  78.55459 ],
       [105.90702 ,  79.22544 ],
       [104.93073 ,  75.02057 ],
       [105.5497  ,  77.008385],
       [105.71745 ,  77.69021 ],
       [105.82806 ,  78.24578 ],
       [104.9742  ,  74.35948 ],
       [105.433914,  75.7614  ],
       [105.52841 ,  76.359055],
       [105.56807 ,  76.82503 ]])

train_pipeline = [
    dict(
        # 把3d target的坐标减去root, 同时删去root, 并保存root_position
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        # 先试试只有一个root
        root_index=0,
        root_name='root_position',
        remove_root=True),
    dict(
        # 添加了target_mean, target_std
        type='NormalizeJointCoordinate',
        item='target',
        mean=joint_3d_normalize_param['mean'],
        std=joint_3d_normalize_param['std']),
    dict(
        # 添加了input_2d_mean, input_2d_std
        type='NormalizeJointCoordinate',
        item='input_2d',
        mean=joint_2d_normalize_param['mean'],
        std=joint_2d_normalize_param['std']),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        # 'root_position', 'root_position_index', 'target_mean', 'target_std'
        # 用来恢复global positions (网络预测的是相对坐标)
        # flip_pairs是干什么的？
        meta_keys=[
            'target_image_path', 'flip_pairs', 'root_position',
            'root_position_index', 'target_mean', 'target_std'
        ])
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='Hand3DH3WBDataset',
        ann_file=f'{data_root}/annotations/train_left_hand.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Hand3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_left_hand.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Hand3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_left_hand.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
