import pickle
from argparse import ArgumentParser
import os.path as osp 
import cv2
import copy
import numpy as np

from mmcv import Config
import mmcv
from mmpose.datasets import DatasetInfo
from mmpose.core import imshow_bboxes, imshow_keypoints


# def get_skeleton():
#     palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
#                         [230, 230, 0], [255, 153, 255], [153, 204, 255],
#                         [255, 102, 255], [255, 51, 255], [102, 178, 255],
#                         [51, 153, 255], [255, 153, 153], [255, 102, 102],
#                         [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                         [51, 255, 51], [0, 255, 0], [0, 0, 255],
#                         [255, 0, 0], [255, 255, 255]])
#     skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
#                         [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
#                         [8, 10], [1, 2], [0, 1], [0, 2],
#                         [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
#                         [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
#                         [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
#                         [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
#                         [102, 103], [91, 104], [104, 105], [105, 106],
#                         [106, 107], [91, 108], [108, 109], [109, 110],
#                         [110, 111], [112, 113], [113, 114], [114, 115],
#                         [115, 116], [112, 117], [117, 118], [118, 119],
#                         [119, 120], [112, 121], [121, 122], [122, 123],
#                         [123, 124], [112, 125], [125, 126], [126, 127],
#                         [127, 128], [112, 129], [129, 130], [130, 131],
#                         [131, 132]]

#     pose_link_color = palette[[
#         0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
#     ] + [16, 16, 16, 16, 16, 16] + [
#         0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
#         16
#     ] + [
#         0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
#         16
#     ]]
#     pose_kpt_color = palette[
#         [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
#         [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]
    
#     return skeleton, pose_kpt_color, pose_link_color


def get_skeleton(args):
    cfg = Config.fromfile(args.config)
    dataset_info = DatasetInfo(cfg.dataset_info)
    skeleton = dataset_info.skeleton
    pose_kpt_color = dataset_info.pose_kpt_color
    pose_link_color = dataset_info.pose_link_color
    return skeleton, pose_kpt_color, pose_link_color


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--pkl-file', required=True)
    parser.add_argument('--video-file', required=True)
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--config', required=True)
    return parser


def main(args):

    skeleton, pose_kpt_color, pose_link_color = get_skeleton(args)
    
    with open(args.pkl_file, 'rb') as fin:
        video_kpts2d_list = pickle.load(fin)
        
    video = mmcv.VideoReader(args.video_file)  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.fps
    
    width, height = video.resolution
    
    
    writer = cv2.VideoWriter(osp.join(args.save_path, f'vis2d_{osp.basename(args.video_file)}'), fourcc, fps, (width, height * 2))
    
    for frame_id, frame_kpts2d_dict in enumerate(mmcv.track_iter_progress(video_kpts2d_list)):
        
        cur_frame = video[frame_id]
        figure = np.zeros((height * 2, width, 3), dtype=np.uint8)
        raw = copy.deepcopy(cur_frame)
        processed = copy.deepcopy(cur_frame)
        del cur_frame
        
        body_bbox = []
        hand_bbox = []
        raw_wholebody_kpts = []
        processed_wholebody_kpts = []
        
        for person in frame_kpts2d_dict.values():
            
            person_bbox = person['person_bbox']
            wholebody_keypoints = person['wholebody_keypoints']
            left_hand_bbox = person['left_hand_bbox']
            left_hand_valid = person['left_hand_valid']
            left_hand_keypoints = person['left_hand_keypoints']
            right_hand_bbox = person['right_hand_bbox']
            right_hand_valid = person['right_hand_valid']
            right_hand_keypoints = person['right_hand_keypoints']
            
            body_bbox.append(person_bbox)
            raw_wholebody_kpts.append(copy.deepcopy(wholebody_keypoints))
            
            if left_hand_valid:
                hand_bbox.append(left_hand_bbox)
                # all_hand_kpts.append(left_hand_keypoints)
                wholebody_keypoints[-42:-21, :] = left_hand_keypoints
                
            if right_hand_valid:
                hand_bbox.append(right_hand_bbox)
                # all_hand_kpts.append(right_hand_keypoints)
                wholebody_keypoints[-21:, :] = right_hand_keypoints
                
            processed_wholebody_kpts.append(wholebody_keypoints)
            
        body_bbox = np.array(body_bbox)
        hand_bbox = np.array(hand_bbox)
        
        # raw
        imshow_bboxes(
            img=raw, 
            bboxes=body_bbox,
            labels=None,
            colors='green',
            text_color='white',
            thickness=3,
            font_scale=1,
            show=False,
            win_name='',
            wait_time=0,
            out_file=None)
        
        imshow_keypoints(
            img=raw,
            pose_result=raw_wholebody_kpts,
            skeleton=skeleton,
            kpt_score_thr=0.0,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            radius=4, 
            thickness=4,
            show_keypoint_weight=False)
        
        # procesed
        imshow_bboxes(
            img=processed, 
            bboxes=body_bbox,
            labels=None,
            colors='green',
            text_color='white',
            thickness=3,
            font_scale=1,
            show=False,
            win_name='',
            wait_time=0,
            out_file=None)
        
        imshow_bboxes(
            img=processed, 
            bboxes=hand_bbox,
            labels=None,
            colors='red',
            text_color='white',
            thickness=3,
            font_scale=1,
            show=False,
            win_name='',
            wait_time=0,
            out_file=None)
            
        imshow_keypoints(
            img=processed,
            pose_result=processed_wholebody_kpts,
            skeleton=skeleton,
            kpt_score_thr=0.0,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            radius=4, 
            thickness=4,
            show_keypoint_weight=False)
        
        figure[:height, :, :] = raw
        figure[height:, :, :] = processed
        
        writer.write(figure)
        
    writer.release()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)