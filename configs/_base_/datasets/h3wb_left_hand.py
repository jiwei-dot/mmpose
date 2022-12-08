dataset_info = dict(
    dataset_name='human3.6m_left_hand',
    paper_info=dict(
        author='',
        title='H3WB: Human3.6M 3D WholeBody Dataset and Benchmark',
        container='',
        year='2022',
        homepage='https://github.com/wholebody3d/wholebody3d',
    ),
    keypoint_info={
        91:
        dict(
            name='left_hand_root',
            id=91,
            color=[255, 255, 255],
            type='',
            swap='right_hand_root'),
        92:
        dict(
            name='left_thumb1',
            id=92,
            color=[255, 128, 0],
            type='',
            swap='right_thumb1'),
        93:
        dict(
            name='left_thumb2',
            id=93,
            color=[255, 128, 0],
            type='',
            swap='right_thumb2'),
        94:
        dict(
            name='left_thumb3',
            id=94,
            color=[255, 128, 0],
            type='',
            swap='right_thumb3'),
        95:
        dict(
            name='left_thumb4',
            id=95,
            color=[255, 128, 0],
            type='',
            swap='right_thumb4'),
        96:
        dict(
            name='left_forefinger1',
            id=96,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger1'),
        97:
        dict(
            name='left_forefinger2',
            id=97,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger2'),
        98:
        dict(
            name='left_forefinger3',
            id=98,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger3'),
        99:
        dict(
            name='left_forefinger4',
            id=99,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger4'),
        100:
        dict(
            name='left_middle_finger1',
            id=100,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger1'),
        101:
        dict(
            name='left_middle_finger2',
            id=101,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger2'),
        102:
        dict(
            name='left_middle_finger3',
            id=102,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger3'),
        103:
        dict(
            name='left_middle_finger4',
            id=103,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger4'),
        104:
        dict(
            name='left_ring_finger1',
            id=104,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger1'),
        105:
        dict(
            name='left_ring_finger2',
            id=105,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger2'),
        106:
        dict(
            name='left_ring_finger3',
            id=106,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger3'),
        107:
        dict(
            name='left_ring_finger4',
            id=107,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger4'),
        108:
        dict(
            name='left_pinky_finger1',
            id=108,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger1'),
        109:
        dict(
            name='left_pinky_finger2',
            id=109,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger2'),
        110:
        dict(
            name='left_pinky_finger3',
            id=110,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger3'),
        111:
        dict(
            name='left_pinky_finger4',
            id=111,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger4'),
    },
    skeleton_info={
        25:
        dict(
            link=('left_hand_root', 'left_thumb1'), id=25, color=[255, 128,
                                                                  0]),
        26:
        dict(link=('left_thumb1', 'left_thumb2'), id=26, color=[255, 128, 0]),
        27:
        dict(link=('left_thumb2', 'left_thumb3'), id=27, color=[255, 128, 0]),
        28:
        dict(link=('left_thumb3', 'left_thumb4'), id=28, color=[255, 128, 0]),
        29:
        dict(
            link=('left_hand_root', 'left_forefinger1'),
            id=29,
            color=[255, 153, 255]),
        30:
        dict(
            link=('left_forefinger1', 'left_forefinger2'),
            id=30,
            color=[255, 153, 255]),
        31:
        dict(
            link=('left_forefinger2', 'left_forefinger3'),
            id=31,
            color=[255, 153, 255]),
        32:
        dict(
            link=('left_forefinger3', 'left_forefinger4'),
            id=32,
            color=[255, 153, 255]),
        33:
        dict(
            link=('left_hand_root', 'left_middle_finger1'),
            id=33,
            color=[102, 178, 255]),
        34:
        dict(
            link=('left_middle_finger1', 'left_middle_finger2'),
            id=34,
            color=[102, 178, 255]),
        35:
        dict(
            link=('left_middle_finger2', 'left_middle_finger3'),
            id=35,
            color=[102, 178, 255]),
        36:
        dict(
            link=('left_middle_finger3', 'left_middle_finger4'),
            id=36,
            color=[102, 178, 255]),
        37:
        dict(
            link=('left_hand_root', 'left_ring_finger1'),
            id=37,
            color=[255, 51, 51]),
        38:
        dict(
            link=('left_ring_finger1', 'left_ring_finger2'),
            id=38,
            color=[255, 51, 51]),
        39:
        dict(
            link=('left_ring_finger2', 'left_ring_finger3'),
            id=39,
            color=[255, 51, 51]),
        40:
        dict(
            link=('left_ring_finger3', 'left_ring_finger4'),
            id=40,
            color=[255, 51, 51]),
        41:
        dict(
            link=('left_hand_root', 'left_pinky_finger1'),
            id=41,
            color=[0, 255, 0]),
        42:
        dict(
            link=('left_pinky_finger1', 'left_pinky_finger2'),
            id=42,
            color=[0, 255, 0]),
        43:
        dict(
            link=('left_pinky_finger2', 'left_pinky_finger3'),
            id=43,
            color=[0, 255, 0]),
        44:
        dict(
            link=('left_pinky_finger3', 'left_pinky_finger4'),
            id=44,
            color=[0, 255, 0]),
        45:
        dict(
            link=('right_hand_root', 'right_thumb1'),
            id=45,
            color=[255, 128, 0]),
        46:
        dict(
            link=('right_thumb1', 'right_thumb2'), id=46, color=[255, 128, 0]),
        47:
        dict(
            link=('right_thumb2', 'right_thumb3'), id=47, color=[255, 128, 0]),
        48:
        dict(
            link=('right_thumb3', 'right_thumb4'), id=48, color=[255, 128, 0]),
        49:
        dict(
            link=('right_hand_root', 'right_forefinger1'),
            id=49,
            color=[255, 153, 255]),
        50:
        dict(
            link=('right_forefinger1', 'right_forefinger2'),
            id=50,
            color=[255, 153, 255]),
        51:
        dict(
            link=('right_forefinger2', 'right_forefinger3'),
            id=51,
            color=[255, 153, 255]),
        52:
        dict(
            link=('right_forefinger3', 'right_forefinger4'),
            id=52,
            color=[255, 153, 255]),
        53:
        dict(
            link=('right_hand_root', 'right_middle_finger1'),
            id=53,
            color=[102, 178, 255]),
        54:
        dict(
            link=('right_middle_finger1', 'right_middle_finger2'),
            id=54,
            color=[102, 178, 255]),
        55:
        dict(
            link=('right_middle_finger2', 'right_middle_finger3'),
            id=55,
            color=[102, 178, 255]),
        56:
        dict(
            link=('right_middle_finger3', 'right_middle_finger4'),
            id=56,
            color=[102, 178, 255]),
        57:
        dict(
            link=('right_hand_root', 'right_ring_finger1'),
            id=57,
            color=[255, 51, 51]),
        58:
        dict(
            link=('right_ring_finger1', 'right_ring_finger2'),
            id=58,
            color=[255, 51, 51]),
        59:
        dict(
            link=('right_ring_finger2', 'right_ring_finger3'),
            id=59,
            color=[255, 51, 51]),
        60:
        dict(
            link=('right_ring_finger3', 'right_ring_finger4'),
            id=60,
            color=[255, 51, 51]),
        61:
        dict(
            link=('right_hand_root', 'right_pinky_finger1'),
            id=61,
            color=[0, 255, 0]),
        62:
        dict(
            link=('right_pinky_finger1', 'right_pinky_finger2'),
            id=62,
            color=[0, 255, 0]),
        63:
        dict(
            link=('right_pinky_finger2', 'right_pinky_finger3'),
            id=63,
            color=[0, 255, 0]),
        64:
        dict(
            link=('right_pinky_finger3', 'right_pinky_finger4'),
            id=64,
            color=[0, 255, 0])
    },
    joint_weights=[1.] * 21,
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066,
        0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031,
        0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045,
        0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
        0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017,
        0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010,
        0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
        0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
        0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
        0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019,
        0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
        0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
        0.019, 0.022, 0.031
    ][91:112])
