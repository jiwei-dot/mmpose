dataset_info = dict(
    dataset_name='',
    paper_info='',
    keypoint_info={
        0:
        dict(name='root', id=0, color=[51, 153, 255], type='lower', swap=''),
        
        1:
        dict(name='right_hip', id=1, color=[255, 128, 0], type='lower', swap='left_hip'),
        2:
        dict(name='right_knee', id=2, color=[255, 128, 0], type='lower', swap='left_knee'),
        3:
        dict(name='right_foot', id=3, color=[255, 128, 0], type='lower', swap='left_foot'),
        
        
        4:
        dict(name='left_hip', id=4, color=[0, 255, 0], type='lower', swap='right_hip'),
        5:
        dict(name='left_knee', id=5, color=[0, 255, 0], type='lower', swap='right_knee'),
        6:
        dict(name='left_foot', id=6, color=[0, 255, 0], type='lower', swap='right_foot'),
        
        7:
        dict(name='spine', id=7, color=[51, 153, 255], type='upper', swap=''),
        8:
        dict(name='thorax', id=8, color=[51, 153, 255], type='upper', swap=''),
        9:
        dict(name='neck_base', id=9, color=[51, 153, 255], type='upper', swap=''),
        10:
        dict(name='head', id=10, color=[51, 153, 255], type='upper', swap=''),
        
        11:
        dict(name='left_shoulder', id=11, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        12:
        dict(name='left_elbow', id=12, color=[0, 255, 0], type='upper', swap='right_elbow'),
        13:
        dict(name='left_wrist', id=13, color=[0, 255, 0], type='upper', swap='right_wrist'),
        
        14:
        dict(name='right_shoulder', id=14, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        15:
        dict(name='right_elbow', id=15, color=[255, 128, 0], type='upper', swap='left_elbow'),
        16:
        dict(name='right_wrist', id=16, color=[255, 128, 0], type='upper', swap='left_wrist'),
        
        17:
        dict(name='left_hand_root', id=17, color=[255, 128, 0], type='', swap='right_hand_root'),   
        18:
        dict(name='left_thumb1', id=18, color=[255, 128, 0], type='', swap='right_thumb1'),
        19:
        dict(name='left_thumb2', id=19, color=[255, 128, 0], type='', swap='right_thumb2'),
        20:
        dict(name='left_thumb3', id=20, color=[255, 128, 0], type='', swap='right_thumb3'),
        21:
        dict(name='left_thumb4', id=21, color=[255, 128, 0], type='', swap='right_thumb4'),
        22:
        dict(name='left_forefinger1', id=22, color=[255, 153, 255], type='', swap='right_forefinger1'),
        23:
        dict(name='left_forefinger2', id=23, color=[255, 153, 255], type='', swap='right_forefinger2'),
        24:
        dict(name='left_forefinger3', id=24, color=[255, 153, 255], type='', swap='right_forefinger3'),
        25:
        dict(name='left_forefinger4', id=25, color=[255, 153, 255], type='', swap='right_forefinger4'),
        26:
        dict(name='left_middle_finger1', id=26, color=[102, 178, 255], type='', swap='right_middle_finger1'),
        27:
        dict(name='left_middle_finger2', id=27, color=[102, 178, 255], type='', swap='right_middle_finger2'),
        28:
        dict(name='left_middle_finger3', id=28, color=[102, 178, 255], type='', swap='right_middle_finger3'),
        29:
        dict(name='left_middle_finger4', id=29, color=[102, 178, 255], type='', swap='right_middle_finger4'),
        30:
        dict(name='left_ring_finger1', id=30, color=[255, 51, 51], type='', swap='right_ring_finger1'),
        31:
        dict(name='left_ring_finger2', id=31, color=[255, 51, 51], type='', swap='right_ring_finger2'),
        32:
        dict(name='left_ring_finger3', id=32, color=[255, 51, 51], type='', swap='right_ring_finger3'),
        33:
        dict(name='left_ring_finger4', id=33, color=[255, 51, 51], type='', swap='right_ring_finger4'),
        34:
        dict(name='left_pinky_finger1', id=34, color=[0, 255, 0], type='', swap='right_pinky_finger1'),
        35:
        dict(name='left_pinky_finger2', id=35, color=[0, 255, 0], type='', swap='right_pinky_finger2'),
        36:
        dict(name='left_pinky_finger3', id=36, color=[0, 255, 0], type='', swap='right_pinky_finger3'),
        37:
        dict(name='left_pinky_finger4', id=37, color=[0, 255, 0], type='', swap='right_pinky_finger4'),
        
        38:
        dict(name='right_hand_root', id=38, color=[255, 128, 0], type='', swap='left_hand_root'),     
        39:
        dict(name='right_thumb1', id=39, color=[255, 128, 0], type='', swap='left_thumb1'),
        40:
        dict(name='right_thumb2', id=40, color=[255, 128, 0], type='', swap='left_thumb2'),
        41:
        dict(name='right_thumb3', id=41, color=[255, 128, 0], type='', swap='left_thumb3'),
        42:
        dict(name='right_thumb4', id=42, color=[255, 128, 0], type='', swap='left_thumb4'),
        43:
        dict(name='right_forefinger1', id=43, color=[255, 153, 255], type='', swap='left_forefinger1'),
        44:
        dict(name='right_forefinger2', id=44, color=[255, 153, 255], type='', swap='left_forefinger2'),
        45:
        dict(name='right_forefinger3', id=45, color=[255, 153, 255], type='', swap='left_forefinger3'),
        46:
        dict(name='right_forefinger4', id=46, color=[255, 153, 255], type='', swap='left_forefinger4'),
        47:
        dict(name='right_middle_finger1', id=47, color=[102, 178, 255], type='', swap='left_middle_finger1'),
        48:
        dict(name='right_middle_finger2', id=48, color=[102, 178, 255], type='', swap='left_middle_finger2'),
        49:
        dict(name='right_middle_finger3', id=49, color=[102, 178, 255], type='', swap='left_middle_finger3'),
        50:
        dict(name='right_middle_finger4', id=50, color=[102, 178, 255], type='', swap='left_middle_finger4'),
        51:
        dict(name='right_ring_finger1', id=51, color=[255, 51, 51], type='', swap='left_ring_finger1'),
        52:
        dict(name='right_ring_finger2', id=52, color=[255, 51, 51], type='', swap='left_ring_finger2'),
        53:
        dict(name='right_ring_finger3', id=53, color=[255, 51, 51], type='', swap='left_ring_finger3'),
        54:
        dict(name='right_ring_finger4', id=54, color=[255, 51, 51], type='', swap='left_ring_finger4'),
        55:
        dict(name='right_pinky_finger1', id=55, color=[0, 255, 0], type='', swap='left_pinky_finger1'),
        56:
        dict(name='right_pinky_finger2', id=56, color=[0, 255, 0], type='', swap='left_pinky_finger2'),
        57:
        dict(name='right_pinky_finger3', id=57, color=[0, 255, 0], type='', swap='left_pinky_finger3'),
        58:
        dict(name='right_pinky_finger4', id=58, color=[0, 255, 0], type='', swap='left_pinky_finger4'),
        
        59:
        dict(name='left_big_toe', id=59, color=[0, 128, 255], type='lower', swap='right_big_toe'),
        60:
        dict(name='left_small_toe', id=60, color=[255, 128, 0], type='lower', swap='right_small_toe'),
        61:
        dict(name='left_heel', id=61, color=[255, 128, 255], type='lower', swap='right_heel'),
        
        62:
        dict(name='right_big_toe', id=62, color=[0, 128, 255], type='lower', swap='left_big_toe'),
        63:
        dict(name='right_small_toe', id=63, color=[255, 128, 0], type='lower', swap='left_small_toe'),
        64:
        dict(name='right_heel', id=64, color=[255, 128, 255], type='lower', swap='left_heel'),
        
    },
    skeleton_info={
        0:
        dict(link=('root', 'left_hip'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_hip', 'left_knee'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_knee', 'left_foot'), id=2, color=[0, 255, 0]),
        
        3:
        dict(link=('root', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_hip', 'right_knee'), id=4, color=[255, 128, 0]),
        5:
        dict(link=('right_knee', 'right_foot'), id=5, color=[255, 128, 0]),
        
        6:
        dict(link=('root', 'spine'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('spine', 'thorax'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('thorax', 'neck_base'), id=8, color=[51, 153, 255]),
        9:
        dict(link=('neck_base', 'head'), id=9, color=[51, 153, 255]),
        
        10:
        dict(link=('thorax', 'left_shoulder'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('left_shoulder', 'left_elbow'), id=11, color=[0, 255, 0]),
        12:
        dict(link=('left_elbow', 'left_wrist'), id=12, color=[0, 255, 0]),
        
        13:
        dict(link=('thorax', 'right_shoulder'), id=13, color=[255, 128, 0]),
        14:
        dict(link=('right_shoulder', 'right_elbow'), id=14, color=[255, 128, 0]),
        15:
        dict(link=('right_elbow', 'right_wrist'), id=15, color=[255, 128, 0]),
        
        16:
        dict(link=('left_hand_root', 'left_thumb1'), id=16, color=[255, 128, 0]),
        17:
        dict(link=('left_thumb1', 'left_thumb2'), id=17, color=[255, 128, 0]),
        18:
        dict(link=('left_thumb2', 'left_thumb3'), id=18, color=[255, 128, 0]),
        19:
        dict(link=('left_thumb3', 'left_thumb4'), id=19, color=[255, 128, 0]),
        20:
        dict(link=('left_hand_root', 'left_forefinger1'), id=20, color=[255, 153, 255]),
        21:
        dict(link=('left_forefinger1', 'left_forefinger2'), id=21, color=[255, 153, 255]),
        22:
        dict(link=('left_forefinger2', 'left_forefinger3'), id=22, color=[255, 153, 255]),
        23:
        dict(link=('left_forefinger3', 'left_forefinger4'), id=23, color=[255, 153, 255]),
        24:
        dict(link=('left_hand_root', 'left_middle_finger1'), id=24, color=[102, 178, 255]),
        25:
        dict(link=('left_middle_finger1', 'left_middle_finger2'), id=25, color=[102, 178, 255]),
        26:
        dict(link=('left_middle_finger2', 'left_middle_finger3'), id=26, color=[102, 178, 255]),
        27:
        dict(link=('left_middle_finger3', 'left_middle_finger4'), id=27, color=[102, 178, 255]),
        28:
        dict(link=('left_hand_root', 'left_ring_finger1'), id=28, color=[255, 51, 51]),
        29:
        dict(link=('left_ring_finger1', 'left_ring_finger2'), id=29, color=[255, 51, 51]),
        30:
        dict(link=('left_ring_finger2', 'left_ring_finger3'), id=30, color=[255, 51, 51]),
        31:
        dict(link=('left_ring_finger3', 'left_ring_finger4'), id=31, color=[255, 51, 51]),
        32:
        dict(link=('left_hand_root', 'left_pinky_finger1'), id=32, color=[0, 255, 0]),
        33:
        dict(link=('left_pinky_finger1', 'left_pinky_finger2'), id=33, color=[0, 255, 0]),
        34:
        dict(link=('left_pinky_finger2', 'left_pinky_finger3'), id=34, color=[0, 255, 0]),
        35:
        dict(link=('left_pinky_finger3', 'left_pinky_finger4'), id=35, color=[0, 255, 0]),
        
        36:
        dict(link=('right_hand_root', 'right_thumb1'), id=36, color=[255, 128, 0]),
        37:
        dict(link=('right_thumb1', 'right_thumb2'), id=37, color=[255, 128, 0]),
        38:
        dict(link=('right_thumb2', 'right_thumb3'), id=38, color=[255, 128, 0]),
        39:
        dict(link=('right_thumb3', 'right_thumb4'), id=39, color=[255, 128, 0]),
        40:
        dict(link=('right_hand_root', 'right_forefinger1'), id=40, color=[255, 153, 255]),
        41:
        dict(link=('right_forefinger1', 'right_forefinger2'), id=41, color=[255, 153, 255]),
        42:
        dict(link=('right_forefinger2', 'right_forefinger3'), id=42, color=[255, 153, 255]),
        43:
        dict(link=('right_forefinger3', 'right_forefinger4'), id=43, color=[255, 153, 255]),
        44:
        dict(link=('right_hand_root', 'right_middle_finger1'), id=44, color=[102, 178, 255]),
        45:
        dict(link=('right_middle_finger1', 'right_middle_finger2'), id=45, color=[102, 178, 255]),
        46:
        dict(link=('right_middle_finger2', 'right_middle_finger3'), id=46, color=[102, 178, 255]),
        47:
        dict(link=('right_middle_finger3', 'right_middle_finger4'), id=47, color=[102, 178, 255]),
        48:
        dict(link=('right_hand_root', 'right_ring_finger1'), id=48, color=[255, 51, 51]),
        49:
        dict(link=('right_ring_finger1', 'right_ring_finger2'), id=49, color=[255, 51, 51]),
        50:
        dict(link=('right_ring_finger2', 'right_ring_finger3'),id=50, color=[255, 51, 51]),
        51:
        dict(link=('right_ring_finger3', 'right_ring_finger4'), id=51, color=[255, 51, 51]),
        52:
        dict(link=('right_hand_root', 'right_pinky_finger1'), id=52, color=[0, 255, 0]),
        53:
        dict(link=('right_pinky_finger1', 'right_pinky_finger2'), id=53, color=[0, 255, 0]),
        54:
        dict(link=('right_pinky_finger2', 'right_pinky_finger3'), id=54, color=[0, 255, 0]),
        55:
        dict(link=('right_pinky_finger3', 'right_pinky_finger4'), id=55, color=[0, 255, 0]),
        
        56:
        dict(link=('left_foot', 'left_big_toe'), id=56, color=[0, 255, 0]),
        57:
        dict(link=('left_foot', 'left_small_toe'), id=57, color=[0, 255, 0]),
        58:
        dict(link=('left_foot', 'left_heel'), id=58, color=[0, 255, 0]),
        
        59:
        dict(link=('right_foot', 'right_big_toe'), id=59, color=[255, 128, 0]),
        60:
        dict(link=('right_foot', 'right_small_toe'), id=60, color=[255, 128, 0]),
        61:
        dict(link=('right_foot', 'right_heel'), id=61, color=[255, 128, 0]),
    },
    joint_weights=[],
    sigmas=[]
)