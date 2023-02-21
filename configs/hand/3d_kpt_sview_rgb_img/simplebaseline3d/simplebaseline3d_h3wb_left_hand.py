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
    mean=[[-0.00012013,  0.00535966,  0.00120815],
       [-0.00024931,  0.00849356,  0.00194111],
       [-0.0003639 ,  0.00996748,  0.00230504],
       [-0.00051297,  0.01550055,  0.0035352 ],
       [-0.00068656,  0.02469996,  0.00559423],
       [-0.00048613,  0.01391142,  0.00307153],
       [-0.00069325,  0.02440946,  0.00541584],
       [-0.00084656,  0.03257335,  0.00726963],
       [-0.00092514,  0.03650211,  0.00816878],
       [-0.00054981,  0.02036143,  0.00446555],
       [-0.00078686,  0.03268817,  0.00723688],
       [-0.00091258,  0.03960425,  0.00881571],
       [-0.00098252,  0.04322392,  0.00965025],
       [-0.00061972,  0.02805167,  0.00616699],
       [-0.00083315,  0.03980802,  0.00880368],
       [-0.00094447,  0.04525442,  0.01005874],
       [-0.00101375,  0.04901246,  0.01092753],
       [-0.0006844 ,  0.03625854,  0.00798757],
       [-0.00083309,  0.04379921,  0.00968393],
       [-0.00090082,  0.04710011,  0.01044457],
       [-0.00098052,  0.05077466,  0.0112977 ]],
    std=[[0.01913914, 0.01635311, 0.02325029],
       [0.03031817, 0.0298881 , 0.03105289],
       [0.04666913, 0.04694139, 0.04690069],
       [0.05849091, 0.06150938, 0.05978256],
       [0.06894936, 0.07389577, 0.0703262 ],
       [0.05612731, 0.06147196, 0.060425  ],
       [0.06846256, 0.07574398, 0.07318217],
       [0.07605363, 0.08236936, 0.08021756],
       [0.0823421 , 0.086531  , 0.08584901],
       [0.0545455 , 0.05954456, 0.05864827],
       [0.06721085, 0.07341718, 0.07163788],
       [0.07407398, 0.07866523, 0.07791986],
       [0.07966381, 0.08282853, 0.08304742],
       [0.05212844, 0.05623996, 0.05570284],
       [0.06255589, 0.06860051, 0.06640315],
       [0.06890661, 0.07269299, 0.07258122],
       [0.0741939 , 0.07615487, 0.07746557],
       [0.04972281, 0.05148521, 0.05224274],
       [0.0573423 , 0.06033872, 0.06054479],
       [0.06130998, 0.06384786, 0.0643785 ],
       [0.06557578, 0.06691035, 0.06888237]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[536.3338864 , 378.28392161],
       [536.31376681, 379.53592823],
       [536.2554731 , 380.22845162],
       [536.20033996, 380.52984723],
       [536.19855542, 381.79564374],
       [536.14866252, 383.92329944],
       [536.44153932, 381.45571156],
       [536.40500051, 383.88602789],
       [536.30627545, 385.78208269],
       [536.23359664, 386.68755868],
       [536.50217209, 382.98484458],
       [536.4145495 , 385.84364222],
       [536.30387969, 387.44625241],
       [536.23411767, 388.27413498],
       [536.48370085, 384.80345957],
       [536.41509797, 387.52718547],
       [536.30918531, 388.78937335],
       [536.2226196 , 389.65558769],
       [536.4515597 , 386.74139074],
       [536.39880592, 388.48946936],
       [536.3355108 , 389.24807265],
       [536.24990643, 390.09817958]],
    std=[[104.66123432,  65.69171794],
       [105.00953662,  67.16412918],
       [105.21382875,  69.46042411],
       [105.51670995,  71.92204116],
       [106.21507753,  74.22630687],
       [106.87097879,  76.35255852],
       [106.9581569 ,  74.16260672],
       [107.72864107,  76.57117383],
       [108.12416419,  77.78374142],
       [108.46523278,  78.59866996],
       [107.36691084,  73.95500135],
       [108.07072208,  76.34728974],
       [108.3858538 ,  77.34246693],
       [108.66813691,  78.09143249],
       [107.4748766 ,  73.51078622],
       [108.10092753,  75.63452756],
       [108.36277125,  76.39913614],
       [108.56381467,  77.00624237],
       [107.53119233,  72.7778257 ],
       [107.96421389,  74.28012353],
       [108.12395574,  74.9397003 ],
       [108.23808729,  75.46382226]])

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