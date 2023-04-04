# 这部分代码需要重构

from argparse import ArgumentParser
import pickle
import numpy as np
import cv2
import copy
import os.path as osp


import mmcv
from mmcv import Config
from mmpose.datasets import DatasetInfo
from mmpose.apis import init_pose_model, inference_pose_lifter_model, vis_3d_pose_result
from mmpose.core import Smoother


# 1. preprocessing kps2d, especially for hands

# 2. 2d kpts lift to 3d

# 3. foot ground contact


def convert_cocowholebody_to_h36m(single_frame_list):
    # single_frame_list: [dict(wholebody_keypoints, track_id), ...]
    new_single_frame_list = []
    for single_person in single_frame_list:
        wholebody_keypoints = single_person['wholebody_keypoints']
        assert tuple(wholebody_keypoints.shape) == (133, 3)
        track_id = single_person['track_id']
        body_keypoints = coco_to_part(wholebody_keypoints, 'body')
        lhand_keypoints = coco_to_part(wholebody_keypoints, 'lhand')
        rhand_keypoints = coco_to_part(wholebody_keypoints, 'rhand')
        lfoot_keypoints = coco_to_part(wholebody_keypoints, 'lfoot')
        rfoot_keypoints = coco_to_part(wholebody_keypoints, 'rfoot')
        keypoints = np.concatenate([body_keypoints, lhand_keypoints, rhand_keypoints, lfoot_keypoints, rfoot_keypoints], axis=0)
        new_single_frame_list.append({
            'h36m_keypoints': keypoints,
            'track_id': track_id
        })
    return new_single_frame_list


def coco_to_part(keypoints, part, extra=True):
    # extra: 手脚部位多包含一个关键点
    assert part in ('body', 'lhand', 'rhand', 'lfoot', 'rfoot')
    if part == 'body':
        keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
        keypoints = keypoints[:17, ...]
        # pelvis (root) is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # in COCO, head is in the middle of l_eye and r_eye
        # in PoseTrack18, head is in the middle of head_bottom and head_top
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new
    
    elif part == 'lhand':
        # indices = [9, ]
        # indices.extend(list(range(91, 112)))
        indices = list(range(91, 112))
        if extra:
            indices.insert(0, 9)
    elif part == 'rhand':
        # indices = [10, ]
        # indices.extend(list(range(112, 133)))
        indices = list(range(112, 133))
        if extra:
            indices.insert(0, 10)
    elif part == 'lfoot':
        # return keypoints[[15, 17, 18, 19], ...]
        indices = [17, 18, 19]
        if extra:
            indices.insert(0, 15)
    elif part == 'rfoot':
        # return keypoints[[16, 20, 21, 22], ...]
        indices = [20, 21, 22]
        if extra:
            indices.insert(0, 16)
    else:
        raise ValueError
    return keypoints[indices, ...]
    
    
def h36m_to_part(keypoints, part):
    assert part in ('body', 'lhand', 'rhand', 'lfoot', 'rfoot')
    if part == 'body':
        indices = list(range(17))
    elif part == 'lhand':
        indices = [13, ]
        indices.extend(list(range(17, 38)))
        return keypoints[indices, ...]
    elif part == 'rhand':
        indices = [16, ]
        indices.extend(list(range(38, 59)))
    elif part == 'lfoot':
        indices = [6, ]
        indices.extend(list(range(59, 62)))
    elif part == 'rfoot':
        indices = [3, ]
        indices.extend(list(range(62, 65)))
    else:
        raise ValueError
    return keypoints[indices, ...]


def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--pkl-path', required=True, help="kpts2d pkl file")
    parser.add_argument('--video-path', required=True)
    parser.add_argument('--save-path', required=True)
    
    parser.add_argument('--device', default='cuda:0')
    
    parser.add_argument('--body3d-lifter-config', default='configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py')
    parser.add_argument('--body3d-lifter-checkpoint', default='workspace/checkpoints/simple3Dbaseline_h36m-f0ad73a4_20210419.pth')
    
    parser.add_argument('--lefthand3d-lifter-config', default='configs/hand/3d_kpt_sview_rgb_img/simplebaseline3d/simplebaseline3d_h3wb_lefthand_local_and_global.py')
    parser.add_argument('--lefthand3d-lifter-checkpoint', default='work_dirs/simplebaseline3d_h3wb_lefthand_local_and_global/best_MPJPE_epoch_180.pth')
    
    parser.add_argument('--righthand3d-lifter-config', default='configs/hand/3d_kpt_sview_rgb_img/simplebaseline3d/simplebaseline3d_h3wb_righthand_local_and_global.py')
    parser.add_argument('--righthand3d-lifter-checkpoint', default='work_dirs/simplebaseline3d_h3wb_righthand_local_and_global/best_MPJPE_epoch_190.pth')
    
    parser.add_argument('--leftfoot3d-lifter-config', default='configs/foot/3d_kpt_sview_rgb_img/simplebaseline3d/simplebaseline3d_h3wb_leftfoot.py')
    parser.add_argument('--leftfoot3d-lifter-checkpoint', default='work_dirs/simplebaseline3d_h3wb_leftfoot/best_MPJPE_epoch_190.pth')
    
    parser.add_argument('--rightfoot3d-lifter-config', default='configs/foot/3d_kpt_sview_rgb_img/simplebaseline3d/simplebaseline3d_h3wb_rightfoot.py')
    parser.add_argument('--rightfoot3d-lifter-checkpoint', default='work_dirs/simplebaseline3d_h3wb_rightfoot/best_MPJPE_epoch_160.pth')
    
    # smooth
    parser.add_argument('--smooth', action='store_true')
    parser.add_argument('--smooth-filter-cfg', default='configs/_base_/filters/smoothnet_t64_h36m.py')
    
    parser.add_argument('--norm-pose-2d', action='store_true')  
    
    parser.add_argument('--complex-process-hand', action='store_true')
    parser.add_argument('--hand-area-thr', default=0.02)
    
    return parser


def init_lift_models(args):
    """
        Init body, lhand, rhand, lfoot, rfoot lifter model.
    """
    body_lift_model = init_pose_model(
        args.body3d_lifter_config,
        args.body3d_lifter_checkpoint,
        device=args.device.lower())
    
    lhand_lift_model = init_pose_model(
        args.lefthand3d_lifter_config,
        args.lefthand3d_lifter_checkpoint,
        device=args.device.lower())
    
    rhand_lift_model = init_pose_model(
        args.righthand3d_lifter_config,
        args.righthand3d_lifter_checkpoint,
        device=args.device.lower())
    
    lfoot_lift_model = init_pose_model(
        args.leftfoot3d_lifter_config,
        args.leftfoot3d_lifter_checkpoint,
        device=args.device.lower())
    
    rfoot_lift_model = init_pose_model(
        args.rightfoot3d_lifter_config,
        args.rightfoot3d_lifter_checkpoint,
        device=args.device.lower())
    
    return body_lift_model, lhand_lift_model, rhand_lift_model, lfoot_lift_model, rfoot_lift_model


def postprocess_hands_inplace(person, pre_frame_id, post_frame_id, video_output_list):
    track_id = person['track_id']
    left_hand_valid = person['left_hand_valid']
    right_hand_valid = person['right_hand_valid']
    
    # left hand
    if left_hand_valid:
        person['wholebody_keypoints'][-42:-21, :] = person['left_hand_keypoints']
    else:
        if video_output_list[pre_frame_id].get(track_id, None) is not None and \
            video_output_list[post_frame_id].get(track_id, None) is not None and \
            video_output_list[pre_frame_id][track_id]['left_hand_valid'] and \
            video_output_list[post_frame_id][track_id]['left_hand_valid']:
                print("use pre and post frame for left hand")
                tmp1 = video_output_list[pre_frame_id][track_id]['left_hand_keypoints']
                tmp2 = video_output_list[post_frame_id][track_id]['left_hand_keypoints']
                person['wholebody_keypoints'][-42:-21, :] = (tmp1 + tmp2) / 2.0
        elif right_hand_valid:
            print((pre_frame_id + post_frame_id) // 2, 'use right hand')
            person['left_hand_keypoints'] = copy.deepcopy(person['right_hand_keypoints'])
            person['left_hand_keypoints'][:, :2] = person['right_hand_keypoints'][:, :2] + person['wholebody_keypoints'][9][:2] - person['wholebody_keypoints'][10][:2]
            c = person['wholebody_keypoints'][9, :2]
            
            # 计算右臂夹角
            vec1 = person['wholebody_keypoints'][10, :2] - person['wholebody_keypoints'][8, :2]
            vec2 = person['right_hand_keypoints'][0, :2] - person['wholebody_keypoints'][10, :2]
            cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            arctan2_ = np.arctan2(sin_, cos_)
            theta1 = arctan2_ / np.pi
            
            # 计算左臂夹角
            vec1 = person['wholebody_keypoints'][9, :2] - person['wholebody_keypoints'][7, :2]
            vec2 = person['left_hand_keypoints'][0, :2] - person['wholebody_keypoints'][9, :2]
            cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            arctan2_ = np.arctan2(sin_, cos_)
            theta2 = arctan2_ / np.pi
            
            r = (theta2 + theta1) * 180
            s = 1
            M = cv2.getRotationMatrix2D(c, r, s)
            person['left_hand_keypoints'][:, :2] = np.column_stack([person['left_hand_keypoints'][:, :2], np.ones(len(person['left_hand_keypoints'][:, :2]))]) @ M.T
            
            # 3. 根据左手镜像翻转
            line_pt = person['wholebody_keypoints'][9, :2]
            line_vec = person['left_hand_keypoints'][0, :2] - line_pt
            for i in range(1, len(person['left_hand_keypoints'])):
                pt_vec = person['left_hand_keypoints'][i, :2] - line_pt
                proj_vec = np.dot(pt_vec, line_vec) / np.dot(line_vec, line_vec) * line_vec
                pt_sym = line_pt + 2 * proj_vec - pt_vec
                person['left_hand_keypoints'][i, :2] = pt_sym
                
            person['wholebody_keypoints'][-42:-21, :] = person['left_hand_keypoints']
            
    # right hand  
    if right_hand_valid:
        person['wholebody_keypoints'][-21:, :] = person['right_hand_keypoints']
    else:
        if video_output_list[pre_frame_id].get(track_id, None) is not None and \
            video_output_list[post_frame_id].get(track_id, None) is not None and \
            video_output_list[pre_frame_id][track_id]['right_hand_valid'] and \
            video_output_list[post_frame_id][track_id]['right_hand_valid']:
                print("use pre and post frame for right hand")
                tmp1 = video_output_list[pre_frame_id][track_id]['right_hand_keypoints']
                tmp2 = video_output_list[post_frame_id][track_id]['right_hand_keypoints']
                person['wholebody_keypoints'][-21:, :] = (tmp1 + tmp2)/ 2.0
        elif left_hand_valid:
            print((pre_frame_id + post_frame_id) // 2, 'use left hand')
            person['right_hand_keypoints'] = copy.deepcopy(person['left_hand_keypoints'])
            person['right_hand_keypoints'][:, :2] = person['left_hand_keypoints'][:, :2] + person['wholebody_keypoints'][10][:2] - person['wholebody_keypoints'][9][:2]  
            c = person['wholebody_keypoints'][10, :2]
            
            # 计算左臂夹角
            vec1 = person['wholebody_keypoints'][9, :2] - person['wholebody_keypoints'][7, :2]
            vec2 = person['left_hand_keypoints'][0, :2] - person['wholebody_keypoints'][9, :2]
            cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            arctan2_ = np.arctan2(sin_, cos_)
            theta1 = arctan2_ / np.pi
            
            # 计算右臂夹角
            vec1 = person['wholebody_keypoints'][10, :2] - person['wholebody_keypoints'][8, :2]
            vec2 = person['right_hand_keypoints'][0, :2] - person['wholebody_keypoints'][10, :2]
            cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
            arctan2_ = np.arctan2(sin_, cos_)
            theta2 = arctan2_ / np.pi
            
            r = (theta2 + theta1) * 180
            s = 1
            M = cv2.getRotationMatrix2D(c, r, s)
            person['right_hand_keypoints'][:, :2] = np.column_stack([person['right_hand_keypoints'][:, :2], np.ones(len(person['right_hand_keypoints'][:, :2]))]) @ M.T
            
            # 3. 根据右手进行镜像翻转
            line_pt = person['wholebody_keypoints'][10, :2]
            line_vec = person['right_hand_keypoints'][0, :2] - line_pt
            for i in range(1, len(person['right_hand_keypoints'])):
                pt_vec = person['right_hand_keypoints'][i, :2] - line_pt
                proj_vec = np.dot(pt_vec, line_vec) / np.dot(line_vec, line_vec) * line_vec
                pt_sym = line_pt + 2 * proj_vec - pt_vec
                person['right_hand_keypoints'][i, :2] = pt_sym
            person['wholebody_keypoints'][-21:, :] = person['right_hand_keypoints']
            
        
def kpts2bbox(kpts, img_w, img_h, scale_factor = 1.2):
    x1 = np.min(kpts[:, 0])
    y1 = np.min(kpts[:, 1])
    x2 = np.max(kpts[:, 0])
    y2 = np.max(kpts[:, 1])
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_w = w * scale_factor
    new_h = h * scale_factor
    new_x1 = max(0, cx - new_w / 2)
    new_y1 = max(0, cy - new_h / 2)
    new_x2 = min(img_w - 1, cx + new_w / 2)
    new_y2 = min(img_h - 1, cy + new_h / 2)
    return np.array([new_x1, new_y1, new_x2, new_y2])

        
# def extract_results_2d(single_frame_list, part, img_w, img_h):
#     assert part in ('body', 'lhand', 'rhand', 'lfoot', 'rfoot')
#     results_2d_list = []
#     for person in single_frame_list:
#         keypoints = h36m_to_part(person['h36m_keypoints'], part)
#         results_2d_list.append({
#             'keypoints': keypoints,
#             'track_id': person['track_id'],
#             'bbox': kpts2bbox(keypoints, img_w, img_h),
#         })
#     return results_2d_list


def extract_results_2d(single_frame_list, part, img_w, img_h):
    assert part in ('body', 'lhand', 'rhand', 'lfoot', 'rfoot')
    results_2d_list = []
    for person in single_frame_list:
        keypoints = person[f'{part}_keypoints']
        results_2d_list.append({
            'keypoints': keypoints,
            'track_id': person['track_id'],
            'bbox': kpts2bbox(keypoints, img_w, img_h),
        })
    return results_2d_list


def merge_kpts_according_trackid(need_merge_kpts_list):
    assert len(need_merge_kpts_list) == 5
    wholebody_wo_face_lift_results = []
    track_ids = [d['track_id'] for d in need_merge_kpts_list[0]]
    
    tmp_need_merge_kpts_list = []
    for part_kpts_list in need_merge_kpts_list:
        tmp_dict = dict()
        for single in part_kpts_list:
            tmp_dict[single['track_id']] = single
        tmp_need_merge_kpts_list.append(tmp_dict)
    
    num_keypoints = 17 + 3 * 2 + 21 * 2
    
    for track_id in track_ids:
        keypoints = np.zeros((num_keypoints, 3))
        keypoints_3d = np.zeros((num_keypoints, 4))
        
        keypoints[:17, ] = tmp_need_merge_kpts_list[0][track_id]['keypoints']
        keypoints[17:38, ] = tmp_need_merge_kpts_list[1][track_id]['keypoints'][1:]
        keypoints[38:59, ] = tmp_need_merge_kpts_list[2][track_id]['keypoints'][1:]
        keypoints[59:62, ] = tmp_need_merge_kpts_list[3][track_id]['keypoints'][1:]
        keypoints[62:65, ] = tmp_need_merge_kpts_list[4][track_id]['keypoints'][1:]
        
        keypoints_3d[:17, ] = tmp_need_merge_kpts_list[0][track_id]['keypoints_3d']
        keypoints_3d[17:38, ] = tmp_need_merge_kpts_list[1][track_id]['keypoints_3d'][1:] + keypoints_3d[13, ]
        keypoints_3d[38:59, ] = tmp_need_merge_kpts_list[2][track_id]['keypoints_3d'][1:] + keypoints_3d[16, ]
        keypoints_3d[59:62, ] = tmp_need_merge_kpts_list[3][track_id]['keypoints_3d'][1:] + keypoints_3d[6, ]
        keypoints_3d[62:65, ] = tmp_need_merge_kpts_list[4][track_id]['keypoints_3d'][1:] + keypoints_3d[3, ]
        
        # 将root点平移到(5, 5, 5)位置方便后续计算相机内参
        keypoints_3d[:, :3] += np.array([5., 5., 5.])
        
        wholebody_wo_face_lift_results.append({
            'track_id': track_id,
            'keypoints': keypoints,
            'keypoints_3d': keypoints_3d
        })
        
    del tmp_need_merge_kpts_list
    return wholebody_wo_face_lift_results


def get_area_of_bbox(bbox):
    if len(bbox) != 0:
        x1, y1, x2, y2, _ = bbox
        return (x2 - x1) * (y2 - y1)
    else:
        return float('inf')


def get_area_of_kpts(kpts):
    x1 = np.min(kpts[:, 0])
    y1 = np.min(kpts[:, 1])
    x2 = np.max(kpts[:, 0])
    y2 = np.max(kpts[:, 1])
    return (x2 - x1) * (y2 - y1)


def replace_lefthand_with_righthand(keypoints_a, keypoints_b):
    # keypoints_a: 133 x 3
    # keypoints_b: 133 x 3
    new_keypoints_a = copy.deepcopy(keypoints_a)
    new_keypoints_a[-42: -21, :2] = keypoints_b[-21:, :2] + keypoints_b[9][:2] - keypoints_b[10][:2]
    c = new_keypoints_a[9, :2]
    
    # 1. 计算右臂夹角
    vec1 = keypoints_b[10, :2] - keypoints_b[8, :2]
    vec2 = keypoints_b[112, :2] - keypoints_b[10, :2]
    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    arctan2_ = np.arctan2(sin_, cos_)
    theta1 = arctan2_ / np.pi
    
    # 2. 计算左臂夹角
    vec1 = new_keypoints_a[9, :2] - new_keypoints_a[7, :2]
    vec2 = new_keypoints_a[91, :2] - new_keypoints_a[9, :2]
    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    arctan2_ = np.arctan2(sin_, cos_)
    theta2 = arctan2_ / np.pi
    
    r = (theta2 + theta1) * 180
    s = 1
    M = cv2.getRotationMatrix2D(c, r, s)
    new_keypoints_a[-42: -21, :2] = np.column_stack([new_keypoints_a[-42: -21, :2], np.ones(len(new_keypoints_a[-42: -21, :2]))]) @ M.T
    
    # 3. 根据左手镜像翻转
    line_pt = new_keypoints_a[9, :2]
    line_vec = new_keypoints_a[91, :2] - line_pt
    for i in range(-42, -21, 1):
        pt_vec = new_keypoints_a[i, :2] - line_pt
        proj_vec = np.dot(pt_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-6) * line_vec
        pt_sym = line_pt + 2 * proj_vec - pt_vec
        new_keypoints_a[i, :2] = pt_sym
        
    # person['wholebody_keypoints'][-42:-21, :] = person['left_hand_keypoints']
    return new_keypoints_a


def replace_righthand_with_lefthand(keypoints_a, keypoints_b):
    new_keypoints_a = copy.deepcopy(keypoints_a)
    new_keypoints_a[-21:, :2] = keypoints_b[-42:-21, :2] + keypoints_b[10][:2] - keypoints_b[9][:2] 
    c = new_keypoints_a[10, :2]
    
    # 1. 计算左臂夹角
    vec1 = keypoints_b[9, :2] - keypoints_b[7, :2]
    vec2 = keypoints_b[91, :2] - keypoints_b[9, :2]
    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    arctan2_ = np.arctan2(sin_, cos_)
    theta1 = arctan2_ / np.pi
    
    # 2. 计算右臂夹角
    vec1 = new_keypoints_a[10, :2] - new_keypoints_a[8, :2]
    vec2 = new_keypoints_a[112, :2] - new_keypoints_a[10, :2]
    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
    arctan2_ = np.arctan2(sin_, cos_)
    theta2 = arctan2_ / np.pi
    
    r = (theta2 + theta1) * 180
    s = 1
    M = cv2.getRotationMatrix2D(c, r, s)
    new_keypoints_a[-21:, :2] = np.column_stack([new_keypoints_a[-21:, :2], np.ones(len(new_keypoints_a[-21:, :2]))]) @ M.T
    
    # 3. 根据右手镜像翻转
    line_pt = new_keypoints_a[10, :2]
    line_vec = new_keypoints_a[112, :2] - line_pt
    for i in range(-21, 0):
        pt_vec = new_keypoints_a[i, :2] - line_pt
        proj_vec = np.dot(pt_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-6) * line_vec
        pt_sym = line_pt + 2 * proj_vec - pt_vec
        new_keypoints_a[i, :2] = pt_sym
    # person['wholebody_keypoints'][-21:, :] = person['right_hand_keypoints']
    return new_keypoints_a


def complex_process_hand_in_single_frame(args, single_frame_dict, video_kpts2d_list, frame_id):
    # single_frame_dict数据结构: dict(track_id:dict('left_hand_bbox', 'left_hand_valid', 'right_hand_bbox', 'right_hand_valid', 'area', 'track_id', 'person_bbox', 'wholebody_keypoints', 'left_hand_keypoints', 'right_hand_keypoints'), ...)
    new_single_frame_dict = copy.deepcopy(single_frame_dict)
    for track_id, person in new_single_frame_dict.items():
        
        wholebody_keypoints = person['wholebody_keypoints']
        person_bbox = person['person_bbox']
        person_box_area = get_area_of_bbox(person_bbox)
        lhand_kpts_area = get_area_of_kpts(wholebody_keypoints[-42:-21, :])
        rhand_kpts_area = get_area_of_kpts(wholebody_keypoints[-21:, :])
        
        left_hand_valid = person['left_hand_valid']
        left_hand_keypoints = person['left_hand_keypoints']
        left_hand_bbox = person['left_hand_bbox']
        lhand_box_area = get_area_of_bbox(left_hand_bbox)
        
        right_hand_valid = person['right_hand_valid']
        right_hand_keypoints = person['right_hand_keypoints']
        right_hand_bbox = person['right_hand_bbox']
        rhand_box_area = get_area_of_bbox(right_hand_bbox)
        
        # 处理左手
        
        # 检测到左手且比例合适
        if left_hand_valid and lhand_box_area / person_box_area <= args.hand_area_thr:
            person['wholebody_keypoints'][-42:-21, :] = left_hand_keypoints
        else:
            # 未检测到左手或检测到但比例不合适
            # 检测到右手且比例合适
            if right_hand_valid and rhand_box_area / person_box_area <= args.hand_area_thr:
                # replace
                keypoints_a = wholebody_keypoints
                keypoints_b = copy.deepcopy(wholebody_keypoints)
                keypoints_b[-21:, :] = right_hand_keypoints
                person['wholebody_keypoints'] = replace_lefthand_with_righthand(keypoints_a, keypoints_b)
            else:
                # 未检测到右手或检测到但比例不合适
                # TcFormer预测左手比例合适,结束考虑
                if lhand_kpts_area / person_box_area <= args.hand_area_thr:
                    pass
                # TcFormer预测右手比例合适
                elif rhand_kpts_area / person_box_area <= args.hand_area_thr:
                    # replace
                    keypoints_a = wholebody_keypoints
                    keypoints_b = copy.deepcopy(wholebody_keypoints)
                    person['wholebody_keypoints'] = replace_lefthand_with_righthand(keypoints_a, keypoints_b)
                else:
                    # use pre frame
                    if frame_id == 0 or video_kpts2d_list[frame_id-1].get(track_id, None) is None:
                        pass
                    else:
                        person['wholebody_keypoints'][-42:-21, :] = video_kpts2d_list[frame_id-1][track_id]['wholebody_keypoints'][-42:-21, :]
                        
                    
                
        # 处理右手
        
        # 检测到右手且比例合适
        if right_hand_valid and rhand_box_area / person_box_area <= args.hand_area_thr:
            person['wholebody_keypoints'][-21:, :] = right_hand_keypoints
        else:
            # 未检测到右手或检测到但比例不合适
            # 检测到左手且比例合适
            if left_hand_valid and lhand_box_area / person_box_area <= args.hand_area_thr:
                # replace
                keypoints_a = wholebody_keypoints
                keypoints_b = copy.deepcopy(wholebody_keypoints)
                keypoints_b[-42:-21, :] = left_hand_keypoints
                person['wholebody_keypoints'] = replace_righthand_with_lefthand(keypoints_a, keypoints_b)
            else:
                # 未检测到左手或检测到但比例不合适
                # TcFormer预测右手比例合适,结束考虑
                if rhand_kpts_area / person_box_area <= args.hand_area_thr:
                    pass
                # TcFormer预测左手比例合适
                elif lhand_kpts_area / person_box_area <= args.hand_area_thr:
                    # replace
                    keypoints_a = wholebody_keypoints
                    keypoints_b = copy.deepcopy(wholebody_keypoints)
                    person['wholebody_keypoints'] = replace_righthand_with_lefthand(keypoints_a, keypoints_b)
                else:
                    # use pre frame
                    if frame_id == 0 or video_kpts2d_list[frame_id-1].get(track_id, None) is None:
                        pass
                    else:
                        person['wholebody_keypoints'][-21:, :] = video_kpts2d_list[frame_id-1][track_id]['wholebody_keypoints'][-21:, :]

    return new_single_frame_dict
            

def simple_process_hand_in_single_frame(args, single_frame_dict, video_kpts2d_list, frame_id):
    new_single_frame_dict = copy.deepcopy(single_frame_dict)
    for track_id, person in new_single_frame_dict.items():
        left_hand_valid = person['left_hand_valid']
        left_hand_keypoints = person['left_hand_keypoints']
        right_hand_valid = person['right_hand_valid']
        right_hand_keypoints = person['right_hand_keypoints']
        if left_hand_valid:
            person['wholebody_keypoints'][-42:-21, :] = left_hand_keypoints
        if right_hand_valid:
            person['wholebody_keypoints'][-21:, :] = right_hand_keypoints
    return new_single_frame_dict


def split_cocowholebody_to_part(single_frame_list):
    data_list = []
    for person in single_frame_list:
        wholebody_keypoints = person['wholebody_keypoints']
        track_id = person['track_id']
        body_keypoints = coco_to_part(wholebody_keypoints, 'body')
        lhand_keypoints = coco_to_part(wholebody_keypoints, 'lhand')
        rhand_keypoints = coco_to_part(wholebody_keypoints, 'rhand')
        lfoot_keypoints = coco_to_part(wholebody_keypoints, 'lfoot')
        rfoot_keypoints = coco_to_part(wholebody_keypoints, 'rfoot')
        data_list.append({
            'track_id': track_id,
            'wholebody_keypoints': wholebody_keypoints,
            'body_keypoints': body_keypoints,
            'lhand_keypoints': lhand_keypoints,
            'rhand_keypoints': rhand_keypoints,
            'lfoot_keypoints': lfoot_keypoints,
            'rfoot_keypoints': rfoot_keypoints
        })
    return data_list


def cal_camera_intrisic(h36m_wo_face_lift_results, track_id=0):
    person_track = None
    for person in h36m_wo_face_lift_results:
        if person['track_id'] == track_id:
            person_track = person
            break
    if person_track is None:
        raise ValueError
    
    kpts_2d = person_track['keypoints'][:17, :2]
    kpts_3d = person_track['keypoints_3d'][:17, :3]

    X = kpts_2d[:, :2]
    y = kpts_3d[:, :2] / (kpts_3d[:, 2:3] + 1e-10)
    
    c1 = (X[:, 0].mean() * (X[:, 0] * y[:, 0]).mean() - (X[:, 0] * X[:, 0]).mean() * (y[:, 0]).mean()) / \
     ((X[:, 0] * y[:, 0]).mean() - X[:, 0].mean() * y[:, 0].mean())

    f1 = ((X[:, 0] * y[:, 0]).mean() - c1 * y[:, 0].mean()) / \
        ((X[:, 0] * X[:, 0]).mean() - 2 * c1 * X[:, 0].mean() + c1 * c1)

    c2 = (X[:, 1].mean() * (X[:, 1] * y[:, 1]).mean() - (X[:, 1] * X[:, 1]).mean() * (y[:, 1]).mean()) / \
        ((X[:, 1] * y[:, 1]).mean() - X[:, 1].mean() * y[:, 1].mean())

    f2 = ((X[:, 1] * y[:, 1]).mean() - c2 * y[:, 1].mean()) / \
        ((X[:, 1] * X[:, 1]).mean() - 2 * c2 * X[:, 1].mean() + c2 * c2)
        
    c = np.array([c1, c2])[np.newaxis, :]
    f = np.array([1.0 / f1, 1.0 / f2])[np.newaxis, :]
    
    return c, f
             

def main(args):
    
    print('*' * 50)
    if args.smooth:
        print("use smoothnet")
    else:
        print("don't use smoothnet")
    if args.complex_process_hand:
        print("complex process hand")
    else:
        print("simple process hand")
    print('*' * 50)
    
    h36m_wo_face_dataset_info = 'configs/_base_/datasets/h36m_wo_face.py'
    cfg = Config.fromfile(h36m_wo_face_dataset_info)
    h36m_wo_face_dataset_info = DatasetInfo(cfg.dataset_info)
    
    with open(args.pkl_path, 'rb') as fin:
        video_kpts2d_list = pickle.load(fin)
        
    video = mmcv.VideoReader(args.video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.fps
    writer = None
    
    assert len(video) == len(video_kpts2d_list)
    
    # 初始化所有部位的二维关键点提升网络
    body_lift_model, lhand_lift_model, rhand_lift_model, \
        lfoot_lift_model, rfoot_lift_model = init_lift_models(args)
        
    smoother_2d = None
    smoother_3d = None
    if args.smooth:
        smoother_2d = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='wholebody_keypoints',
            keypoint_dim=2)
        
        # smoother_3d = Smoother(
        #     filter_cfg=args.smooth_filter_cfg,
        #     keypoint_key='body_keypoints_3d',
        #     keypoint_dim=3)
    
    video_output_list = []
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        
        single_frame_dict = video_kpts2d_list[frame_id]
        
        if args.complex_process_hand:
            single_frame_dict = complex_process_hand_in_single_frame(args, single_frame_dict, video_kpts2d_list, frame_id)
        else:
            single_frame_dict = simple_process_hand_in_single_frame(args, single_frame_dict, video_kpts2d_list, frame_id)
            # # postprocess hands, using pre and post frame, using another hand
            # for track_id, person in single_frame_dict.items():
            #     postprocess_hands_inplace(person, pre_frame_id, post_frame_id, video_kpts2d_list)    
            
        # track_id用在手部处理，现在不需要了
        single_frame_list = []
        for track_id, person in single_frame_dict.items():
            single_frame_list.append({
                    'wholebody_keypoints': person['wholebody_keypoints'],
                    'track_id': person['track_id']
                })
            
        # print(single_frame_list)
            
        # 将coco-wholebody分解成五个部位
        # single_frame_list数据结构:[dict(track_id, wholebody_keypoints, body_keypoints, lhand_keypoints, rhand_keypoints, lfoot_keypoints, rfoot_keypoints), ...]
        single_frame_list = split_cocowholebody_to_part(single_frame_list)

        # # before: single_frame_list: [dict(wholebody_keypoints, track_id), ...]
        # single_frame_list = convert_cocowholebody_to_h36m(single_frame_list)
        # # after: single_frame_list: [dict(h36m_keypoints, track_id), ...]
            
        if smoother_2d is not None:
            single_frame_list = smoother_2d.smooth(single_frame_list)
            if smoother_2d.key == 'wholebody_keypoints':
                single_frame_list = split_cocowholebody_to_part(single_frame_list)
                   
        # extract different part sequence
        # xxx_results_2d数据结构: [dict(keypoints, track_id, bbox), ...]
        body_results_2d = extract_results_2d(single_frame_list, 'body', video.width, video.height)
        lhand_results_2d = extract_results_2d(single_frame_list, 'lhand', video.width, video.height)
        rhand_results_2d = extract_results_2d(single_frame_list, 'rhand', video.width, video.height)
        lfoot_results_2d = extract_results_2d(single_frame_list, 'lfoot', video.width, video.height)
        rfoot_results_2d = extract_results_2d(single_frame_list, 'rfoot', video.width, video.height)
        
        # body二维关键点提升
        # body_lift_results: [dict(track_id, keypoints(NxKx3), keypoints_3d(Kx4)), ...]
        pre_body_results_2d_dict = dict()
        for person in body_results_2d:
            pre_body_results_2d_dict[person['track_id']] = copy.deepcopy(person)
        body_lift_results = inference_pose_lifter_model(
            body_lift_model,
            pose_results_2d=[body_results_2d],
            dataset='Body3DH36MDataset',
            dataset_info=DatasetInfo(body_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        for person in body_lift_results:
            person['keypoints'] = pre_body_results_2d_dict[person['track_id']]['keypoints']
        del pre_body_results_2d_dict

        
        # lhand二维关键点提升
        # lhand_lift_results: [dict(track_id, keypoints(NxKx3), keypoints_3d(Kx4)), ...]
        pre_lhand_results_2d_dict = dict()
        for person in lhand_results_2d:
            pre_lhand_results_2d_dict[person['track_id']] = copy.deepcopy(person)
        lhand_lift_results = inference_pose_lifter_model(
            lhand_lift_model,
            pose_results_2d=[lhand_results_2d],
            dataset='Hand3DH3WBDataset',
            dataset_info=DatasetInfo(lhand_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        for person in lhand_lift_results:
            person['keypoints'] = pre_lhand_results_2d_dict[person['track_id']]['keypoints']
        del pre_lhand_results_2d_dict
        
        # rhand二维关键点提升
        # rhand_lift_results: [dict(track_id, keypoints(NxKx3), keypoints_3d(Kx4)), ...]
        pre_rhand_results_2d_dict = dict()
        for person in rhand_results_2d:
            pre_rhand_results_2d_dict[person['track_id']] = copy.deepcopy(person)
        rhand_lift_results = inference_pose_lifter_model(
            rhand_lift_model,
            pose_results_2d=[rhand_results_2d],
            dataset='Hand3DH3WBDataset',
            dataset_info=DatasetInfo(rhand_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        for person in rhand_lift_results:
            person['keypoints'] = pre_rhand_results_2d_dict[person['track_id']]['keypoints']
        del pre_rhand_results_2d_dict
        
        # lfoot二维关键点提升
        # lfoot_lift_results: [dict(track_id, keypoints(NxKx3), keypoints_3d(Kx4)), ...]
        pre_lfoot_results_2d_dict = dict()
        for person in lfoot_results_2d:
            pre_lfoot_results_2d_dict[person['track_id']] = copy.deepcopy(person)
        lfoot_lift_results = inference_pose_lifter_model(
            lfoot_lift_model,
            pose_results_2d=[lfoot_results_2d],
            dataset='Foot3DH3WBDataset',
            dataset_info=DatasetInfo(lfoot_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        for person in lfoot_lift_results:
            person['keypoints'] = pre_lfoot_results_2d_dict[person['track_id']]['keypoints']
        del pre_lfoot_results_2d_dict
        
        # rfoot二维关键点提升
        # rfoot_lift_results: [dict(track_id, keypoints(NxKx3), keypoints_3d(Kx4)), ...]
        pre_rfoot_results_2d_dict = dict()
        for person in rfoot_results_2d:
            pre_rfoot_results_2d_dict[person['track_id']] = copy.deepcopy(person)
        rfoot_lift_results = inference_pose_lifter_model(
            rfoot_lift_model,
            pose_results_2d=[rfoot_results_2d],
            dataset='Foot3DH3WBDataset',
            dataset_info=DatasetInfo(rfoot_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        for person in rfoot_lift_results:
            person['keypoints'] = pre_rfoot_results_2d_dict[person['track_id']]['keypoints']
        del pre_rfoot_results_2d_dict
        
        assert len(body_lift_results) == len(lhand_lift_results) == len(rhand_lift_results) == \
            len(lfoot_lift_results) == len(rfoot_lift_results)
        
        # h36m_wo_face_lift_results数据结构:[dict(track_id, keypoints, keypoints_3d), ...]
        h36m_wo_face_lift_results = merge_kpts_according_trackid([body_lift_results, lhand_lift_results, 
                                                                rhand_lift_results, lfoot_lift_results, rfoot_lift_results])
            
        # 3d smooth
        if smoother_3d is not None:
            
            for person in h36m_wo_face_lift_results:
                person['body_keypoints_3d'] = person['keypoints_3d'][:17]
            
            h36m_wo_face_lift_results = smoother_3d.smooth(h36m_wo_face_lift_results)
            
            for person in h36m_wo_face_lift_results:
                person['keypoints_3d'][:17] = person['body_keypoints_3d']
                del person['body_keypoints_3d']
                
        # 这里的代码有些问题，第一帧必须要有人
        if frame_id == 0:
            intrisic_c, intrisic_f = cal_camera_intrisic(h36m_wo_face_lift_results, track_id=0)
        else:
            for person in h36m_wo_face_lift_results:
                root_2d = person['keypoints'][0:1][:, :2]
                root_3d = np.array([0., 0., 5.])
                root_3d[:2] = (root_2d - intrisic_c) / intrisic_f * root_3d[2]
                person['keypoints_3d'][:, :3] += root_3d - np.array([[5., 5., 5.]])
                
    
        video_output_list.append(copy.deepcopy(h36m_wo_face_lift_results))
        
        # post process for better visualize
        # for person in h36m_wo_face_lift_results:
        #     keypoints_3d = person['keypoints_3d']
        #     keypoints_3d = keypoints_3d[..., [0, 2, 1]]
        #     keypoints_3d[..., 2] = -keypoints_3d[..., 2]
        #     # keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)
        #     person['keypoints_3d'] = keypoints_3d
        
        if len(h36m_wo_face_lift_results) == 0:
            continue
        
        img_vis = vis_3d_pose_result(
            body_lift_model,
            result=h36m_wo_face_lift_results,
            img=cur_frame,
            dataset_info=h36m_wo_face_dataset_info,
            kpt_score_thr=0,
            out_file=None,
            radius=5,
            thickness=5,
            vis_height=1000,
            num_instances=-1,
            show=False,
            axis_azimuth=-90)
        
        # cv2.imwrite(f'{frame_id}.jpg', img_vis)
        # if frame_id > 2:
        #     exit()
        
        if writer is None:
            writer = cv2.VideoWriter(
                osp.join(args.save_path, f'vis_{osp.basename(args.video_path)}'), fourcc, fps, (img_vis.shape[1], img_vis.shape[0]))
        writer.write(img_vis)
    
    writer.release()
    tmp = osp.basename(args.video_path).split('.')[0]
    name = f'video_{tmp}_kpts3d.pkl'
    with open(osp.join(args.save_path, name), 'wb') as fout:
        pickle.dump(video_output_list, fout)
        

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)