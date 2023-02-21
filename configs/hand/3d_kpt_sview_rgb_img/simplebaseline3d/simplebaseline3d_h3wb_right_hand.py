# right_hand: coco annotations [112, 132]

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
    mean=[[-5.62176372e-05,  3.54513024e-03,  7.73103566e-04],
       [-1.08868699e-04,  3.61886625e-03,  7.13261897e-04],
       [-1.30730947e-04,  1.74801129e-03,  2.27510544e-04],
       [-2.49210501e-04,  5.56983053e-03,  1.02801760e-03],
       [-3.89601720e-04,  1.26227428e-02,  2.56732024e-03],
       [-2.84432483e-04,  4.86774952e-03,  9.62948235e-04],
       [-4.44654108e-04,  1.26385669e-02,  2.64803252e-03],
       [-5.46869555e-04,  1.98373429e-02,  4.23249806e-03],
       [-5.82091379e-04,  2.25882915e-02,  4.81554821e-03],
       [-3.82905066e-04,  1.13593579e-02,  2.44781975e-03],
       [-5.54646668e-04,  2.10119223e-02,  4.55376264e-03],
       [-6.26441306e-04,  2.70217940e-02,  5.86983329e-03],
       [-6.71227038e-04,  3.02015004e-02,  6.54803919e-03],
       [-4.71086916e-04,  1.93227502e-02,  4.26241501e-03],
       [-6.44147352e-04,  2.88244359e-02,  6.33250557e-03],
       [-6.98439911e-04,  3.35816815e-02,  7.37160680e-03],
       [-7.38653719e-04,  3.70683564e-02,  8.13212046e-03],
       [-5.67143924e-04,  2.72515345e-02,  6.05986816e-03],
       [-6.65339532e-04,  3.31328440e-02,  7.34160902e-03],
       [-7.05783575e-04,  3.62202190e-02,  8.00588289e-03],
       [-7.40877872e-04,  3.98590572e-02,  8.81140912e-03]],
    std=[[0.01634911, 0.01719761, 0.0188622 ],
       [0.02833186, 0.03125015, 0.02992937],
       [0.04417503, 0.04847328, 0.04856937],
       [0.05820659, 0.06386409, 0.06460439],
       [0.06975725, 0.07717844, 0.07689707],
       [0.05496381, 0.06339935, 0.06187324],
       [0.06897879, 0.07875859, 0.07675289],
       [0.07562061, 0.08651273, 0.08330052],
       [0.08225354, 0.09176655, 0.08896735],
       [0.05358296, 0.06213864, 0.05854543],
       [0.06715437, 0.07718329, 0.0728263 ],
       [0.07380243, 0.08362528, 0.07891832],
       [0.07995137, 0.08895008, 0.08457226],
       [0.05085085, 0.05881293, 0.0540012 ],
       [0.06338947, 0.0726907 , 0.067493  ],
       [0.06913121, 0.07752497, 0.07264021],
       [0.07389171, 0.08168147, 0.07728612],
       [0.0487543 , 0.05386037, 0.05079207],
       [0.0574467 , 0.06397073, 0.05988872],
       [0.06183113, 0.06798978, 0.06402954],
       [0.06506304, 0.0719923 , 0.06751961]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[532.01477644, 369.4778058 ],
       [531.99211628, 370.31139549],
       [532.10941133, 370.28285639],
       [532.22127021, 369.80185555],
       [532.2878117 , 370.66679867],
       [532.33716899, 372.3021973 ],
       [532.08411404, 370.53497478],
       [532.15425463, 372.33586632],
       [532.21106041, 374.00902164],
       [532.29116277, 374.62889265],
       [532.00666305, 372.07658375],
       [532.08713308, 374.32105114],
       [532.16037197, 375.70946845],
       [532.23718149, 376.42437009],
       [531.94206933, 373.97326574],
       [532.02769753, 376.18116121],
       [532.09528442, 377.27892474],
       [532.15075321, 378.07800741],
       [531.89085478, 375.86703136],
       [531.95943955, 377.22787489],
       [532.02730368, 377.93492247],
       [532.05895432, 378.77287025]],
    std=[[133.44174545,  69.75603225],
       [133.6803469 ,  71.33491955],
       [133.08055115,  73.57591806],
       [132.74820342,  75.93234149],
       [132.92112282,  78.34887033],
       [133.17421395,  80.53974004],
       [135.39476797,  78.63451786],
       [135.55886098,  81.10663918],
       [135.36084235,  82.33382927],
       [134.99794643,  83.1281857 ],
       [136.12976148,  78.6227839 ],
       [136.14185415,  81.02846347],
       [135.75953857,  81.97671759],
       [135.21425571,  82.70750099],
       [136.49648778,  78.17087828],
       [136.38921095,  80.35026491],
       [135.97289962,  81.02314109],
       [135.60187956,  81.58024421],
       [136.63384653,  77.34150113],
       [136.59702607,  78.92871087],
       [136.21405341,  79.50526377],
       [135.98201016,  80.04288857]])

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
        ann_file=f'{data_root}/annotations/train_right_hand.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Hand3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_right_hand.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Hand3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_right_hand.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)