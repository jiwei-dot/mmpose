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
    mean=[[-3.0620882e-05,  1.1083345e-03,  2.0074565e-04],
       [-3.4572251e-05,  2.1289762e-04, -4.3710261e-05],
       [-1.1678368e-04,  4.8867785e-03,  9.6571713e-04],
       [-2.2623035e-04,  1.2579678e-02,  2.6644790e-03],
       [-1.2457019e-04,  3.6272069e-03,  7.0753263e-04],
       [-2.4853103e-04,  1.2273946e-02,  2.6116804e-03],
       [-3.4025850e-04,  2.0002063e-02,  4.3349089e-03],
       [-3.7008672e-04,  2.3173695e-02,  5.0332416e-03],
       [-2.0505271e-04,  9.7071845e-03,  2.0856711e-03],
       [-3.4438845e-04,  2.0305000e-02,  4.4317599e-03],
       [-4.1302716e-04,  2.6818981e-02,  5.8826813e-03],
       [-4.5207606e-04,  3.0412247e-02,  6.6783279e-03],
       [-2.9042189e-04,  1.7240731e-02,  3.7954729e-03],
       [-4.2715642e-04,  2.7434083e-02,  6.0512479e-03],
       [-4.8196319e-04,  3.2788463e-02,  7.2462452e-03],
       [-5.1969825e-04,  3.6678594e-02,  8.1132520e-03],
       [-3.7879849e-04,  2.4732461e-02,  5.4961089e-03],
       [-4.5940490e-04,  3.1187821e-02,  6.9238530e-03],
       [-4.9343030e-04,  3.4577213e-02,  7.6754005e-03],
       [-5.3010095e-04,  3.8501956e-02,  8.5548237e-03]],
    std=[[0.01955715, 0.01923858, 0.02258341],
       [0.03752273, 0.03792159, 0.04356057],
       [0.05198287, 0.05374113, 0.0604429 ],
       [0.06343495, 0.067298  , 0.07298886],
       [0.04692999, 0.05355477, 0.05631301],
       [0.06124554, 0.06915212, 0.07196821],
       [0.06799016, 0.07688887, 0.07882259],
       [0.07464434, 0.08207419, 0.08467245],
       [0.04505148, 0.05231384, 0.05288251],
       [0.05892814, 0.06762861, 0.06805655],
       [0.06566658, 0.07392798, 0.07436883],
       [0.07198295, 0.07934383, 0.08040419],
       [0.04213447, 0.04900258, 0.04828174],
       [0.0551537 , 0.06312455, 0.06295381],
       [0.06105793, 0.06786864, 0.06832555],
       [0.06582735, 0.07209938, 0.07302863],
       [0.04050785, 0.04418428, 0.04551587],
       [0.04944332, 0.05448496, 0.05540372],
       [0.0539661 , 0.05848229, 0.05979258],
       [0.05721902, 0.06252428, 0.06332556]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[534.39795, 364.25577],
       [534.4909 , 364.46234],
       [534.58887, 364.2066 ],
       [534.64545, 365.26416],
       [534.6812 , 367.04807],
       [534.50494, 364.98495],
       [534.55334, 366.99017],
       [534.5837 , 368.79382],
       [534.63715, 369.516  ],
       [534.43695, 366.4311 ],
       [534.4905 , 368.89996],
       [534.53143, 370.4134 ],
       [534.58044, 371.2313 ],
       [534.37427, 368.23633],
       [534.4308 , 370.60776],
       [534.46826, 371.85486],
       [534.50616, 372.75116],
       [534.3213 , 370.038  ],
       [534.3663 , 371.53433],
       [534.4116 , 372.3193 ],
       [534.42456, 373.22592]],
    std=[[128.81541 ,  75.12151 ],
       [128.13847 ,  77.35797 ],
       [127.73427 ,  79.68793 ],
       [127.8864  ,  82.10849 ],
       [128.13939 ,  84.29981 ],
       [130.33315 ,  82.596054],
       [130.4913  ,  85.05202 ],
       [130.29932 ,  86.22422 ],
       [129.96245 ,  86.957275],
       [131.12474 ,  82.63126 ],
       [131.14331 ,  85.017975],
       [130.76404 ,  85.87966 ],
       [130.25415 ,  86.550476],
       [131.55458 ,  82.19569 ],
       [131.45862 ,  84.34688 ],
       [131.07533 ,  84.94863 ],
       [130.71019 ,  85.4631  ],
       [131.76326 ,  81.36872 ],
       [131.74146 ,  82.945274],
       [131.36966 ,  83.47075 ],
       [131.15576 ,  83.95964 ]])

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
