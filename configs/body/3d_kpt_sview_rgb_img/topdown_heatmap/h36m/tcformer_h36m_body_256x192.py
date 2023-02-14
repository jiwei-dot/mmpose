_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py',
]


# 采用什么评价标准?
evaluation = dict(interval=1, metric=['mpjpe', 'p-mpjpe', 'n-mpjpe'], save_best='MPJPE')


optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200],
)
total_epochs = 210


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)


channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)),
)


norm_cfg = dict(
    typw='SyncBN',
    requires_grad=True
)


model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/tcformer-4e1adbf1_20220421.pth',
    backbone=dict(
        type='TCFormer',
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        num_layers=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1),
    neck=dict(
        type='MTA',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[4, 4, 4, 4],
        num_outs=4,
        use_sr_conv=False,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleDoubleHeads',
        in_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1,),
        loss_keypoint_list=[
            dict(type='JointsMSELoss', use_target_weight=True),
            dict(type='JointsMSELoss', use_target_weight=True),
        ]     
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,        
        post_process='default',
        shift_heatmap=False,
        modulate_kernel=11
    )
)

data_root = 'data/h36m'
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    
    # soft_nms=False,
    # nms_thr=1.0,
    # oks_thr=0.9,
    # vis_thr=0.2,
    # use_gt_bbox=False,
    # det_bbox_thr=0.0,
    # bbox_file=None,
    
    # results['target_2d']
    need_2d_label=True,
    
    # results['camera_param'], results['image_width'], results['image_height']
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)


train_pipeline = [
    dict(type='RenameKeys', key_pairs=[('target_image_path', 'image_file'),]),
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type="CustomGenerateJoints"),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    # dict(
    #     type='TopDownHalfBodyTransform', 
    #     num_joints_half_body=8, 
    #     prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', 
        rot_factor=40, 
        scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    
    
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(type='CustomGenerateXYZInCameraSpace'),
    
    
    # img:               3 x H x W
    # target:            N x h x w
    # target_weight:     N x 1
    # xyz_camera_space:  N x 3
    # xyz_weight:        N x 1
    
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight', 'xyz_camera_space', 'xyz_weight'],
        meta_keys=[
            'image_file', 
            'joints_3d', 'joints_3d_visible',
            'joints_4d', 'joints_4d_visible',
            'camera_param',
            'center', 'scale', 
            'rotation',
            # 'bbox_score',
            'flip_pairs'
        ]
    )
]


val_pipeline = [
    dict(type='RenameKeys', key_pairs=[('target_image_path', 'image_file'),]),
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type="CustomGenerateJoints"),
    dict(
        type='TopDownGetRandomScaleRotation', 
        rot_factor=0, 
        scale_factor=0),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 
            'joints_3d', 'joints_3d_visible',
            'joints_4d', 'joints_4d_visible',
            'camera_param',
            'center', 'scale', 
            'rotation',
            # 'bbox_score',
            'flip_pairs'
        ]
    )
]

test_pipeline = val_pipeline


data_root = 'data/h36m'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='Body2D3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type='Body2D3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    test=dict(
        type='Body2D3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
)