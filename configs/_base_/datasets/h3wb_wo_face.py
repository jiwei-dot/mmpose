dataset_info = dict(
    dataset_name='human3.6m_wholebody',
    paper_info=dict(
        author='',
        title='H3WB: Human3.6M 3D WholeBody Dataset and Benchmark',
        container='',
        year='2022',
        homepage='https://github.com/wholebody3d/wholebody3d',
    ),
    keypoint_info={
        0:
        dict(name='hip', id=0, color=[0, 255, 0], type='lower', swap=''),
        1:
        dict(name='nose', id=1, color=[51, 153, 255], type='upper', swap=''),
        2:
        dict(
            name='left_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        3:
        dict(
            name='right_eye',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        4:
        dict(
            name='left_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        5:
        dict(
            name='right_ear',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        6:
        dict(
            name='left_shoulder',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        7:
        dict(
            name='right_shoulder',
            id=7,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        8:
        dict(
            name='left_elbow',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        9:
        dict(
            name='right_elbow',
            id=9,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        10:
        dict(
            name='left_wrist',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        11:
        dict(
            name='right_wrist',
            id=11,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        12:
        dict(
            name='left_hip',
            id=12,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        13:
        dict(
            name='right_hip',
            id=13,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        14:
        dict(
            name='left_knee',
            id=14,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        15:
        dict(
            name='right_knee',
            id=15,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        16:
        dict(
            name='left_ankle',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        17:
        dict(
            name='right_ankle',
            id=17,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        18:
        dict(
            name='left_big_toe',
            id=18,
            color=[255, 128, 0],
            type='lower',
            swap='right_big_toe'),
        19:
        dict(
            name='left_small_toe',
            id=19,
            color=[255, 128, 0],
            type='lower',
            swap='right_small_toe'),
        20:
        dict(
            name='left_heel',
            id=20,
            color=[255, 128, 0],
            type='lower',
            swap='right_heel'),
        21:
        dict(
            name='right_big_toe',
            id=21,
            color=[255, 128, 0],
            type='lower',
            swap='left_big_toe'),
        22:
        dict(
            name='right_small_toe',
            id=22,
            color=[255, 128, 0],
            type='lower',
            swap='left_small_toe'),
        23:
        dict(
            name='right_heel',
            id=23,
            color=[255, 128, 0],
            type='lower',
            swap='left_heel'),
        24:
        dict(
            name='left_hand_root',
            id=24,
            color=[255, 255, 255],
            type='',
            swap='right_hand_root'),
        25:
        dict(
            name='left_thumb1',
            id=25,
            color=[255, 128, 0],
            type='',
            swap='right_thumb1'),
        26:
        dict(
            name='left_thumb2',
            id=26,
            color=[255, 128, 0],
            type='',
            swap='right_thumb2'),
        27:
        dict(
            name='left_thumb3',
            id=27,
            color=[255, 128, 0],
            type='',
            swap='right_thumb3'),
        28:
        dict(
            name='left_thumb4',
            id=28,
            color=[255, 128, 0],
            type='',
            swap='right_thumb4'),
        29:
        dict(
            name='left_forefinger1',
            id=29,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger1'),
        30:
        dict(
            name='left_forefinger2',
            id=30,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger2'),
        31:
        dict(
            name='left_forefinger3',
            id=31,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger3'),
        32:
        dict(
            name='left_forefinger4',
            id=32,
            color=[255, 153, 255],
            type='',
            swap='right_forefinger4'),
        33:
        dict(
            name='left_middle_finger1',
            id=33,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger1'),
        34:
        dict(
            name='left_middle_finger2',
            id=34,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger2'),
        35:
        dict(
            name='left_middle_finger3',
            id=35,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger3'),
        36:
        dict(
            name='left_middle_finger4',
            id=36,
            color=[102, 178, 255],
            type='',
            swap='right_middle_finger4'),
        37:
        dict(
            name='left_ring_finger1',
            id=37,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger1'),
        38:
        dict(
            name='left_ring_finger2',
            id=38,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger2'),
        39:
        dict(
            name='left_ring_finger3',
            id=39,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger3'),
        40:
        dict(
            name='left_ring_finger4',
            id=40,
            color=[255, 51, 51],
            type='',
            swap='right_ring_finger4'),
        41:
        dict(
            name='left_pinky_finger1',
            id=41,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger1'),
        42:
        dict(
            name='left_pinky_finger2',
            id=42,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger2'),
        43:
        dict(
            name='left_pinky_finger3',
            id=43,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger3'),
        44:
        dict(
            name='left_pinky_finger4',
            id=44,
            color=[0, 255, 0],
            type='',
            swap='right_pinky_finger4'),
        45:
        dict(
            name='right_hand_root',
            id=45,
            color=[255, 255, 255],
            type='',
            swap='left_hand_root'),
        46:
        dict(
            name='right_thumb1',
            id=46,
            color=[255, 128, 0],
            type='',
            swap='left_thumb1'),
        47:
        dict(
            name='right_thumb2',
            id=47,
            color=[255, 128, 0],
            type='',
            swap='left_thumb2'),
        48:
        dict(
            name='right_thumb3',
            id=48,
            color=[255, 128, 0],
            type='',
            swap='left_thumb3'),
        49:
        dict(
            name='right_thumb4',
            id=49,
            color=[255, 128, 0],
            type='',
            swap='left_thumb4'),
        50:
        dict(
            name='right_forefinger1',
            id=50,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger1'),
        51:
        dict(
            name='right_forefinger2',
            id=51,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger2'),
        52:
        dict(
            name='right_forefinger3',
            id=52,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger3'),
        53:
        dict(
            name='right_forefinger4',
            id=53,
            color=[255, 153, 255],
            type='',
            swap='left_forefinger4'),
        54:
        dict(
            name='right_middle_finger1',
            id=54,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger1'),
        55:
        dict(
            name='right_middle_finger2',
            id=55,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger2'),
        56:
        dict(
            name='right_middle_finger3',
            id=56,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger3'),
        57:
        dict(
            name='right_middle_finger4',
            id=57,
            color=[102, 178, 255],
            type='',
            swap='left_middle_finger4'),
        58:
        dict(
            name='right_ring_finger1',
            id=58,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger1'),
        59:
        dict(
            name='right_ring_finger2',
            id=59,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger2'),
        60:
        dict(
            name='right_ring_finger3',
            id=60,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger3'),
        61:
        dict(
            name='right_ring_finger4',
            id=61,
            color=[255, 51, 51],
            type='',
            swap='left_ring_finger4'),
        62:
        dict(
            name='right_pinky_finger1',
            id=62,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger1'),
        63:
        dict(
            name='right_pinky_finger2',
            id=63,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger2'),
        64:
        dict(
            name='right_pinky_finger3',
            id=64,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger3'),
        65:
        dict(
            name='right_pinky_finger4',
            id=65,
            color=[0, 255, 0],
            type='',
            swap='left_pinky_finger4')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
        19:
        dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
        20:
        dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
        21:
        dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
        22:
        dict(
            link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
        23:
        dict(
            link=('right_ankle', 'right_small_toe'),
            id=23,
            color=[255, 128, 0]),
        24:
        dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
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
    joint_weights=[1.] * 66,
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[])
