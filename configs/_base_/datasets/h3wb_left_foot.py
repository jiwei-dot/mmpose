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
        dict(name='left_ankle', id=0, color=[0, 255, 0], type='lower', swap=''),
        1:
        dict(name='left_big_toe', id=1, color=[255, 0, 0], type='lower', swap=''),
        2:
        dict(name='left_small_toe', id=2, color=[0, 255, 0], type='lower', swap=''),
        3:
        dict(name='left_heel', id=3, color=[0, 0, 255], type='lower', swap=''),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_big_toe'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_ankle', 'left_small_toe'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_ankle', 'left_heel'), id=2, color=[0, 255, 0]),
    },
    joint_weights=[1.] * 4,
    sigmas=[],
    stats_info=dict(
        bbox_center=[533.02689777, 584.97778603],
        bbox_scale=0.1703109830795288,
    )
)
