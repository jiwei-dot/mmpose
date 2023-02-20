import pickle
import cv2
import os
import math
import numpy as np
from argparse import ArgumentParser

import mmcv
from mmpose.datasets import DatasetInfo
from configs._base_.datasets.coco_wholebody_hand import dataset_info


hand_info = DatasetInfo(dataset_info)

        
def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--pkl-path')
    parser.add_argument('--video-path')
    parser.add_argument('--save-root')
    
    return parser


def cv_plot(kpts1, kpts2, bbox, image):
    assert kpts1.shape == kpts2.shape
    x1, y1, x2, y2, _ = bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
    skeleton = hand_info.skeleton
    image = imshow_keypoints(image, kpts1, skeleton, (0, 0, 255))
    image = imshow_keypoints(image, kpts2, skeleton, (0, 255, 0))
    return image
    
    
def imshow_keypoints(img, kpts, skeleton, color, thickness=1, radius=1, show_keypoint_weight=False):
    
    img_h, img_w, _ = img.shape
    
    # draw keypoints
    for kpt in kpts:
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if show_keypoint_weight:
            img_copy = img.copy()
            cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius, color, -1, cv2.LINE_AA)
            transparency = max(0, min(1, kpt_score))
            cv2.addWeighted(
                img_copy,
                transparency,
                img,
                1 - transparency,
                0,
                dst=img)
        else:
            cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1, cv2.LINE_AA)
    
    # draw skeletons
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
            or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
            or pos2[1] <= 0 or pos2[1] >= img_h):
            continue
        
        if show_keypoint_weight:
            img_copy = img.copy()
            X = (pos1[0], pos2[0])
            Y = (pos1[1], pos2[1])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            stickwidth = 1
            polygon = cv2.ellipse2Poly(
                (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                int(angle), 0, 360, 1)
            cv2.fillConvexPoly(img_copy, polygon, color)
            transparency = max(
                0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
            cv2.addWeighted(
                img_copy,
                transparency,
                img,
                1 - transparency,
                0,
                dst=img)
        else:
            cv2.line(img, pos1, pos2, color, thickness=thickness, lineType=cv2.LINE_AA)
            
    return img


def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    video = mmcv.VideoReader(args.video_path)
    
    with open(args.pkl_path, 'rb') as fin:
        results_list = pickle.load(fin)
        
    assert len(video) == len(results_list)
    
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        show = False
        results = results_list[frame_id]
        if len(results) == 0:
            continue
        for result in results:
            wholebody_keypoints = result['wholebody_keypoints']
            if result['left_hand_valid']:
                show = True
                lh_from_wholebody = wholebody_keypoints[91: 112, ]
                lh_from_lh = result['left_hand_keypoints']
                lh_bbox = result['left_hand_bbox']
                cur_frame = cv_plot(lh_from_wholebody, lh_from_lh, lh_bbox, cur_frame)
                
            if result['right_hand_valid']:
                show = True
                rh_from_wholebody = wholebody_keypoints[112: 133, ]
                rh_from_rh = result['right_hand_keypoints']
                rh_bbox = result['right_hand_bbox']
                cur_frame = cv_plot(rh_from_wholebody, rh_from_rh, rh_bbox, cur_frame)
                
        if show:
            cv2.imwrite(os.path.join(args.save_root, f'{frame_id}.jpg'), cur_frame)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)