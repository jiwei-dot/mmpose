_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco_wholebody_hand.py',
]



evaluation = dict(interval=1, metric=['PCK', 'AUC', 'MPJPE'], save_best='MPJPE')


optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
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

# checkpoint
load_from = "workspace/checkpoints/modified_res50_cocowholebody_hand_256x256.pth"


channel_cfg = dict(
    num_output_channels=21,
    dataset_joints=21,
    dataset_channel=[list(range(21))],
    inference_channel=list(range(21)),
)


# model setting
model = dict(
    type='CustomTopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TopdownHeatmapDoubleHeads',
        in_channels=2048,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint_list=[
            dict(type='JointsMSELoss', use_target_weight=True),
            dict(type='L1Loss', use_target_weight=True),
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


data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    # dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', 
        rot_factor=0, 
        scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(type='RenameKeys', key_pairs=[('target', 'target_xy'), 
                                       ('target_weight', 'target_xy_weight')]),
    dict(type='CustomGenerateDepth'),
    dict(
        type='Collect',
        keys=['img', 'target_xy', 'target_xy_weight', 'target_z', 'target_z_weight'],
        meta_keys=[
            'image_file', 
            'joints_3d', 'joints_3d_visible',
            'joints_4d', 'joints_4d_visible',
            # 'camera_param',
            'center', 'scale', 
            'rotation',
            'bbox_score',
            'abs_depth',
            # 'flip_pairs'
        ]
    )
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
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
    dict(type='CustomGenerateDepth'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 
            'joints_3d', 'joints_3d_visible',
            'joints_4d', 'joints_4d_visible',
            # 'camera_param',
            'center', 'scale', 
            'rotation',
            'bbox_score',
            'abs_depth',
            # 'flip_pairs'
        ]
    )
]

test_pipeline = val_pipeline


data_root = 'data/merge_cocohand_and_h3wbhand'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='Hand2DAnd3DMergeDataset',
        ann_file=f'{data_root}/annotations/merge_cocohand_and_h3wbhand_train_v1.0.json',
        img_prefix=f'{data_root}/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
        test_mode=False,
    ),
    val=dict(
        type='Hand2DAnd3DMergeDataset',
        ann_file=f'{data_root}/annotations/merge_cocohand_and_h3wbhand_val_v1.0.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
        test_mode=True,
    ),
    test=dict(
        type='Hand2DAnd3DMergeDataset',
        ann_file=f'{data_root}/annotations/merge_cocohand_and_h3wbhand_val_v1.0.json',
        img_prefix=f'{data_root}/val/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
        test_mode=True,
    ),
)