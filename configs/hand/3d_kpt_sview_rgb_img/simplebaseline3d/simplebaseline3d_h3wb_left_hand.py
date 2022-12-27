# left_hand: coco annotations [91, 111]

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
    num_output_channels=22,
    dataset_joints=22,
    dataset_channel=[
       list(range(22)),
    ],
    inference_channel=list(range(22)))

# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 22,
        stem_channels=1024,
        num_blocks=3,
        kernel_sizes=(1, 1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=22-1,  # do not predict root joint (both left and right)
        loss_keypoint=dict(type='MSELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/h3wb'
data_cfg = dict(
    num_joints=22,
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
    mean=[[-0.00011263,  0.00642347,  0.00143905],
       [-0.00022911,  0.01168701,  0.00262102],
       [-0.00033108,  0.01531904,  0.00343359],
       [-0.0004595 ,  0.02279937,  0.00509641],
       [-0.00061396,  0.03348505,  0.0074849 ],
       [-0.0004182 ,  0.02110059,  0.0046692 ],
       [-0.00060156,  0.03338899,  0.00741362],
       [-0.00072659,  0.04216569,  0.00938778],
       [-0.00078181,  0.04617826,  0.01029528],
       [-0.000466  ,  0.02702681,  0.00598116],
       [-0.00067265,  0.0408663 ,  0.00908145],
       [-0.00077287,  0.04812346,  0.01071841],
       [-0.00081909,  0.05180334,  0.01155399],
       [-0.00052897,  0.03389005,  0.00751874],
       [-0.00071487,  0.04694764,  0.01044403],
       [-0.00079563,  0.05255895,  0.01171346],
       [-0.0008453 ,  0.05637191,  0.01257875],
       [-0.00058799,  0.04104308,  0.00912649],
       [-0.00071393,  0.04942178,  0.01100236],
       [-0.00076258,  0.05286195,  0.01178108],
       [-0.00082106,  0.05662127,  0.01263382]],
    std=[[0.02033005, 0.01805403, 0.02385847],
       [0.03144161, 0.0309425 , 0.03172385],
       [0.04779397, 0.04761484, 0.04775052],
       [0.05934392, 0.06184696, 0.06016785],
       [0.06935462, 0.07402381, 0.07020639],
       [0.05625869, 0.06241482, 0.0606136 ],
       [0.06797669, 0.07630076, 0.07261669],
       [0.07479014, 0.08250308, 0.07907737],
       [0.08023685, 0.08625852, 0.08411247],
       [0.0539505 , 0.06047764, 0.05816107],
       [0.06602974, 0.07393152, 0.07054039],
       [0.07204151, 0.07869229, 0.07616674],
       [0.07654428, 0.08247196, 0.08074453],
       [0.0515109 , 0.05725754, 0.05512158],
       [0.06119388, 0.06913411, 0.06519498],
       [0.06662873, 0.07280735, 0.07082146],
       [0.0711067 , 0.07598581, 0.07525894],
       [0.04901524, 0.05254573, 0.05163673],
       [0.05628496, 0.0610322 , 0.05963104],
       [0.0595683 , 0.06427713, 0.06308338],
       [0.06314862, 0.06701928, 0.06724472]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[537.20570859, 377.12434049],
       [537.17771188, 378.62842602],
       [537.11320122, 379.81811799],
       [537.05846506, 380.62222982],
       [537.03423538, 382.34471781],
       [536.96701051, 384.82098769],
       [537.20970292, 381.97976525],
       [537.14822617, 384.82871118],
       [537.05897568, 386.86609646],
       [536.99105871, 387.79024683],
       [537.24170622, 383.38765485],
       [537.13878578, 386.59931101],
       [537.04227566, 388.27978434],
       [536.98387847, 389.1209448 ],
       [537.21406292, 385.0153305 ],
       [537.12908073, 388.04278236],
       [537.04194028, 389.3407462 ],
       [536.96744196, 390.21751562],
       [537.17429975, 386.7082137 ],
       [537.11757782, 388.65210893],
       [537.06148509, 389.44165403],
       [536.99058846, 390.30840933]],
    std=[[101.7063113 ,  67.34878119],
       [102.24106577,  68.92682377],
       [102.45413563,  71.03556404],
       [102.76944623,  73.31745501],
       [103.51769479,  75.47357773],
       [104.2056635 ,  77.4878276 ],
       [104.36893097,  75.55384741],
       [105.16485552,  77.83154289],
       [105.5021098 ,  78.94510738],
       [105.76050722,  79.67587943],
       [104.79733647,  75.40271548],
       [105.514887  ,  77.66050142],
       [105.74932083,  78.55475595],
       [105.90770959,  79.22544463],
       [104.93111983,  75.0206518 ],
       [105.55008599,  77.00859821],
       [105.71786091,  77.69075484],
       [105.82888526,  78.2462793 ],
       [104.97466661,  74.35977823],
       [105.43456715,  75.76167864],
       [105.52902567,  76.35929672],
       [105.56853451,  76.82536375]])

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
    samples_per_gpu=512,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=512),
    test_dataloader=dict(samples_per_gpu=512),
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