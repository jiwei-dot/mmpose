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
        dict(name='right_ankle', id=0, color=[255, 128, 0], type='lower', swap=''),
        1:
        dict(name='right_big_toe', id=1, color=[255, 0, 0], type='lower', swap=''),
        2:
        dict(name='right_small_toe', id=2, color=[0, 255, 0], type='lower', swap=''),
        3:
        dict(name='right_heel', id=3, color=[0, 0, 255], type='lower', swap=''),
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_big_toe'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('right_ankle', 'right_small_toe'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('right_ankle', 'right_heel'), id=2, color=[255, 128, 0]),
    },
    joint_weights=[1.] * 4,
    sigmas=[])
