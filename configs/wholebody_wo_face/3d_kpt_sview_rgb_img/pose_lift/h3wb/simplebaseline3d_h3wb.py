_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h3wb_wo_face.py'
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

# channel_cfg = dict(
#     num_output_channels=66,
#     dataset_joints=66,
#     dataset_channel=[
#        list(range(66)),
#     ],
#     inference_channel=list(range(66)))


# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TCN',
        in_channels=2 * 66,
        stem_channels=1024,
        num_blocks=4,
        kernel_sizes=(1, 1, 1, 1, 1),
        dropout=0.5),
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=66-1,  # do not predict root joint
        loss_keypoint=dict(type='MSELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))

# data settings
data_root = 'data/h3wb'
data_cfg = dict(
    num_joints=66,
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
    mean=[[ 7.08793729e-03, -5.91103505e-01, -1.33363690e-01],
       [ 7.53402586e-03, -6.19897049e-01, -1.39848324e-01],
       [ 7.36461970e-03, -6.16660532e-01, -1.39042231e-01],
       [ 7.69972961e-03, -6.11492176e-01, -1.37809810e-01],
       [ 7.32904561e-03, -6.07051304e-01, -1.36647631e-01],
       [ 5.88666589e-03, -4.44589888e-01, -1.00243568e-01],
       [ 5.21684517e-03, -4.42427617e-01, -9.93695676e-02],
       [ 3.54319205e-03, -2.41693903e-01, -5.48681401e-02],
       [ 2.64052899e-03, -2.57789389e-01, -5.76661563e-02],
       [ 2.27436156e-03, -1.66303283e-01, -3.81485062e-02],
       [ 2.09146976e-03, -2.22604771e-01, -5.00443564e-02],
       [ 2.20325685e-04, -4.51187674e-04, -2.46544050e-04],
       [-2.20325648e-04,  4.51187286e-04,  2.46543991e-04],
       [-4.26952308e-03,  3.33352660e-01,  7.44968485e-02],
       [-4.71221046e-03,  3.31290922e-01,  7.45141008e-02],
       [-9.06106072e-03,  7.13662626e-01,  1.60009537e-01],
       [-9.47250725e-03,  7.11637550e-01,  1.60248016e-01],
       [-1.00631068e-02,  7.66946765e-01,  1.71595029e-01],
       [-9.89483821e-03,  7.69648829e-01,  1.72262450e-01],
       [-9.51706910e-03,  7.53967341e-01,  1.69166307e-01],
       [-1.05439785e-02,  7.66024973e-01,  1.72286713e-01],
       [-1.04962834e-02,  7.66523630e-01,  1.72513849e-01],
       [-9.91303247e-03,  7.53085828e-01,  1.69640865e-01],
       [ 2.16172839e-03, -1.59879817e-01, -3.67094595e-02],
       [ 2.04525408e-03, -1.54616271e-01, -3.55274824e-02],
       [ 1.94328057e-03, -1.50984242e-01, -3.47149160e-02],
       [ 1.81486341e-03, -1.43503913e-01, -3.30520926e-02],
       [ 1.66040145e-03, -1.32818228e-01, -3.06636065e-02],
       [ 1.85615680e-03, -1.45202694e-01, -3.34793104e-02],
       [ 1.67279769e-03, -1.32914296e-01, -3.07348866e-02],
       [ 1.54776696e-03, -1.24137593e-01, -2.87607278e-02],
       [ 1.49254888e-03, -1.20125028e-01, -2.78532243e-02],
       [ 1.80836196e-03, -1.39276472e-01, -3.21673477e-02],
       [ 1.60171422e-03, -1.25436978e-01, -2.90670572e-02],
       [ 1.50148860e-03, -1.18179825e-01, -2.74300987e-02],
       [ 1.45527306e-03, -1.14499948e-01, -2.65945187e-02],
       [ 1.74538968e-03, -1.32413234e-01, -3.06297626e-02],
       [ 1.55948836e-03, -1.19355646e-01, -2.77044780e-02],
       [ 1.47873503e-03, -1.13744329e-01, -2.64350418e-02],
       [ 1.42906263e-03, -1.09931369e-01, -2.55697548e-02],
       [ 1.68637112e-03, -1.25260199e-01, -2.90220138e-02],
       [ 1.56043116e-03, -1.16881504e-01, -2.71461496e-02],
       [ 1.51177842e-03, -1.13441329e-01, -2.63674302e-02],
       [ 1.45330542e-03, -1.09682013e-01, -2.55146842e-02],
       [ 2.06018461e-03, -2.19265656e-01, -4.93172703e-02],
       [ 2.02956386e-03, -2.18157323e-01, -4.91165249e-02],
       [ 2.02561281e-03, -2.19052755e-01, -4.93609794e-02],
       [ 1.94340101e-03, -2.14378901e-01, -4.83515533e-02],
       [ 1.83395491e-03, -2.06685972e-01, -4.66528030e-02],
       [ 1.93561413e-03, -2.15638469e-01, -4.86097359e-02],
       [ 1.81165482e-03, -2.06991712e-01, -4.67055780e-02],
       [ 1.71992647e-03, -1.99263595e-01, -4.49823445e-02],
       [ 1.69009663e-03, -1.96091995e-01, -4.42840087e-02],
       [ 1.85513259e-03, -2.09558504e-01, -4.72316023e-02],
       [ 1.71579485e-03, -1.98960595e-01, -4.48854939e-02],
       [ 1.64715801e-03, -1.92446870e-01, -4.34345986e-02],
       [ 1.60810914e-03, -1.88853399e-01, -4.26389657e-02],
       [ 1.76976280e-03, -2.02024918e-01, -4.55218242e-02],
       [ 1.63302936e-03, -1.91831519e-01, -4.32659960e-02],
       [ 1.57822084e-03, -1.86477142e-01, -4.20710245e-02],
       [ 1.54048585e-03, -1.82586929e-01, -4.12039719e-02],
       [ 1.68138701e-03, -1.94533083e-01, -4.38211741e-02],
       [ 1.60078064e-03, -1.88077785e-01, -4.23934108e-02],
       [ 1.56675463e-03, -1.84688347e-01, -4.16418830e-02],
       [ 1.53008504e-03, -1.80763913e-01, -4.07624463e-02]],
    std=[[0.12041315, 0.08134075, 0.17206625],
       [0.11724582, 0.07888904, 0.16388964],
       [0.11555527, 0.07862584, 0.16314287],
       [0.10356109, 0.06210726, 0.12164781],
       [0.09908386, 0.06144798, 0.11957064],
       [0.14027293, 0.05142466, 0.11040465],
       [0.13511646, 0.05377899, 0.11566072],
       [0.22942646, 0.1047121 , 0.17152056],
       [0.23368959, 0.12707005, 0.19426329],
       [0.25790487, 0.20051036, 0.21837022],
       [0.26409147, 0.2264602 , 0.24616802],
       [0.08640148, 0.01271649, 0.0544487 ],
       [0.08640148, 0.01271649, 0.0544487 ],
       [0.13780506, 0.13944428, 0.19064105],
       [0.13739674, 0.14042895, 0.19450951],
       [0.1524277 , 0.15551212, 0.20669579],
       [0.15997832, 0.15697456, 0.20337462],
       [0.19991026, 0.16000219, 0.27855963],
       [0.20633971, 0.15690227, 0.25957754],
       [0.14945915, 0.15726498, 0.20245879],
       [0.20287085, 0.15941206, 0.27458757],
       [0.2122838 , 0.15682289, 0.26002063],
       [0.15861943, 0.15829703, 0.19871381],
       [0.26229108, 0.21109054, 0.22608421],
       [0.26091197, 0.22338129, 0.22900239],
       [0.26042174, 0.23650123, 0.23399671],
       [0.26573273, 0.24880152, 0.24173204],
       [0.27007531, 0.2597636 , 0.24766238],
       [0.27892972, 0.25126363, 0.25193277],
       [0.28398077, 0.26360814, 0.25908334],
       [0.28419698, 0.26881872, 0.26118397],
       [0.28408168, 0.27174018, 0.26190486],
       [0.28497029, 0.25079836, 0.2545255 ],
       [0.28833666, 0.26269172, 0.26087517],
       [0.2873993 , 0.26635045, 0.26133544],
       [0.28643522, 0.26884044, 0.26104146],
       [0.28723065, 0.24859812, 0.25451689],
       [0.29036054, 0.25909413, 0.25943399],
       [0.28868998, 0.26170925, 0.25973814],
       [0.28724987, 0.26365181, 0.25928726],
       [0.28870573, 0.24468621, 0.25276358],
       [0.2907369 , 0.25216727, 0.25682592],
       [0.28960455, 0.25460946, 0.256664  ],
       [0.28757486, 0.25632336, 0.25672032],
       [0.26800391, 0.23625802, 0.25198695],
       [0.26852878, 0.24856884, 0.25729482],
       [0.2704589 , 0.26118152, 0.26433448],
       [0.27587004, 0.27408855, 0.27385098],
       [0.28120101, 0.28557443, 0.28197487],
       [0.28798359, 0.27749449, 0.28168579],
       [0.29342871, 0.29027263, 0.29068115],
       [0.2948678 , 0.29594899, 0.29376491],
       [0.29527274, 0.29920123, 0.29538701],
       [0.29196884, 0.27781509, 0.2833604 ],
       [0.29656903, 0.29004549, 0.29139388],
       [0.29677659, 0.29397843, 0.29285132],
       [0.2961986 , 0.29688394, 0.29381097],
       [0.29320123, 0.27563054, 0.28241548],
       [0.29700433, 0.28671503, 0.28966846],
       [0.29698753, 0.28929216, 0.29057001],
       [0.29648568, 0.29141847, 0.29134125],
       [0.2933929 , 0.27142052, 0.28101911],
       [0.29624489, 0.27947802, 0.28590829],
       [0.29588537, 0.2818342 , 0.28669062],
       [0.2956846 , 0.28392673, 0.28716336]])

# 2D joint normalization parameters
# From file: '{data_root}/annotations/joint2d_stats.pkl'
joint_2d_normalize_param = dict(
    mean=[[534.66762418, 415.94097806],
       [537.86366476, 274.72341024],
       [538.1932465 , 267.70031816],
       [537.59537973, 268.39129848],
       [538.28075123, 269.97504486],
       [537.07592461, 270.80948129],
       [537.7418793 , 310.82859537],
       [535.64816885, 310.97836433],
       [537.20229894, 359.44873687],
       [534.22419403, 355.29109886],
       [537.20570859, 377.12434049],
       [534.3937687 , 363.47234316],
       [535.30477118, 415.81671133],
       [534.03047714, 416.06524468],
       [534.9281039 , 492.64959733],
       [532.9747506 , 492.51100934],
       [532.90338803, 576.78355853],
       [531.54909992, 576.87212007],
       [533.56692915, 588.95513987],
       [533.65039025, 589.21284974],
       [532.47219793, 585.468885  ],
       [531.52146222, 589.35963141],
       [531.06593207, 589.46362616],
       [531.3871354 , 585.76267864],
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
       [536.99058846, 390.30840933],
       [534.39510799, 364.25681917],
       [534.4933386 , 364.46391216],
       [534.58987611, 364.20328317],
       [534.64678549, 365.26259269],
       [534.6834311 , 367.04525486],
       [534.50283717, 364.98623415],
       [534.55402435, 366.98962615],
       [534.58710392, 368.79223115],
       [534.6407263 , 369.51561666],
       [534.43697941, 366.43346887],
       [534.48989301, 368.90070871],
       [534.53120039, 370.41464728],
       [534.58225603, 371.23242017],
       [534.37137678, 368.23661734],
       [534.43032743, 370.60772338],
       [534.46898805, 371.85364785],
       [534.50318501, 372.75212922],
       [534.32161689, 370.03645283],
       [534.36595246, 371.53353072],
       [534.41184416, 372.31567971],
       [534.42406108, 373.22573619]],
    std=[[100.2977205 ,  47.26532585],
       [101.67127659,  57.25374246],
       [100.46790251,  57.6319743 ],
       [103.86591017,  57.66317539],
       [ 98.74171457,  56.50981104],
       [107.0222862 ,  56.72249889],
       [ 96.87798059,  52.91421368],
       [114.45551744,  53.36231971],
       [ 99.37887021,  55.94132102],
       [127.00589781,  58.96965229],
       [101.7063113 ,  67.34878119],
       [128.54919768,  73.45864342],
       [ 95.85638518,  47.21738245],
       [108.33827623,  47.41548444],
       [100.30782639,  43.10121808],
       [110.40126528,  43.62703338],
       [102.99067186,  50.39087227],
       [111.80010231,  52.13738756],
       [103.53227537,  51.7706836 ],
       [102.80298566,  51.19201958],
       [103.84026609,  51.53965411],
       [112.69958285,  53.80829198],
       [116.15619428,  54.00611491],
       [111.78637547,  53.28889042],
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
       [105.56853451,  76.82536375],
       [128.81592638,  75.12171169],
       [128.1384109 ,  77.35841358],
       [127.73455312,  79.68801433],
       [127.88663746,  82.10860454],
       [128.14003232,  84.30004735],
       [130.33338753,  82.59647594],
       [130.49162679,  85.05207275],
       [130.29975398,  86.22462966],
       [129.96267062,  86.95770314],
       [131.12536778,  82.63172133],
       [131.14352892,  85.01801221],
       [130.76468123,  85.8797951 ],
       [130.25454323,  86.55113112],
       [131.55500054,  82.19627789],
       [131.45910056,  84.3471309 ],
       [131.07592095,  84.94891121],
       [130.71074298,  85.46381101],
       [131.76341848,  81.36914561],
       [131.74181651,  82.94548306],
       [131.37058021,  83.47086548],
       [131.15626107,  83.9599305 ]])

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
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64),
    train=dict(
        type='WholeBody3DH3WBDataset',
        ann_file=f'{data_root}/annotations/train_whole_body_wo_face.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='WholeBody3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_whole_body_wo_face.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='WholeBody3DH3WBDataset',
        ann_file=f'{data_root}/annotations/val_whole_body_wo_face.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
