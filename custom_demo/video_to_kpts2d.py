from argparse import ArgumentParser
import copy
import pickle
import os

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_pose_model, process_mmdet_results, inference_top_down_pose_model, get_track_id
from mmpose.datasets import DatasetInfo


def assign_handbbox_for_eachperson(kpt_thr, old_data_list, hand_bboxes):
    """
        For every person_dict in old_data_list, assign left and right hand.
        This will add `left(right)_hand_bbox` and `left(right)_hand_valid` keys.
    """
    # old_data_list: [dict(bbox, keypoints, area, track_id), ...],为单张图片的wholebody2d检测结果, 长度对应单张图片检测到的人数
    # hand_bboxes: ndarray [Nx5]
    data_list = copy.deepcopy(old_data_list)
    num_hand_bboxes = len(hand_bboxes)
    if num_hand_bboxes == 0:
        for idx in range(len(data_list)):
            data_list[idx]['left_hand_bbox'] = np.empty((0, 5))
            data_list[idx]['left_hand_valid'] = False
            data_list[idx]['right_hand_bbox'] = np.empty((0, 5))
            data_list[idx]['right_hand_valid'] = False
    else:
        for idx, data in enumerate(data_list):  
            dist_left_arm = np.ones((num_hand_bboxes, )) * float('inf')
            dist_right_arm = np.ones((num_hand_bboxes, )) * float('inf')
            person_keypoints2d = data['keypoints']
            
            # left arm
            if person_keypoints2d[9][-1] > kpt_thr and person_keypoints2d[7][-1] > kpt_thr:
                dist_wrist_elbow = np.linalg.norm(person_keypoints2d[9][:2] - person_keypoints2d[7][:2])
                c_x = (hand_bboxes[:, 0] + hand_bboxes[:, 2]) / 2
                c_y = (hand_bboxes[:, 1] + hand_bboxes[:, 3]) / 2
                center = np.stack((c_x, c_y), axis=-1)
                dist_bbox_ankle = np.linalg.norm(center - person_keypoints2d[9][:2], axis=-1)
                mask = dist_bbox_ankle < dist_wrist_elbow * 1.5
                dist_left_arm[mask] = dist_bbox_ankle[mask]

            # right arm
            if person_keypoints2d[10][-1] > kpt_thr and person_keypoints2d[8][-1] > kpt_thr:
                dist_wrist_elbow = np.linalg.norm(person_keypoints2d[10][:2] - person_keypoints2d[8][:2])
                c_x = (hand_bboxes[:, 0] + hand_bboxes[:, 2]) / 2
                c_y = (hand_bboxes[:, 1] + hand_bboxes[:, 3]) / 2
                center = np.stack((c_x, c_y), axis=-1)
                dist_bbox_ankle = np.linalg.norm(center - person_keypoints2d[10][:2], axis=-1)
                mask = dist_bbox_ankle < dist_wrist_elbow * 1.5
                dist_right_arm[mask] = dist_bbox_ankle[mask]
                
            left_id = np.argmin(dist_left_arm)
            right_id = np.argmin(dist_right_arm)
            
            if left_id != right_id:
                if dist_left_arm[left_id] < float('inf'):
                    data_list[idx]['left_hand_bbox'] = hand_bboxes[left_id]
                    data_list[idx]['left_hand_valid'] = True
                else:
                    data_list[idx]['left_hand_bbox'] = np.empty((0, 5))
                    data_list[idx]['left_hand_valid'] = False
                    
                if dist_right_arm[right_id] < float('inf'):
                    data_list[idx]['right_hand_bbox'] = hand_bboxes[right_id]
                    data_list[idx]['right_hand_valid'] = True
                else:
                    data_list[idx]['right_hand_bbox'] = np.empty((0, 5))
                    data_list[idx]['right_hand_valid'] = False 
            else:
                assign_hand = None
                assign_id = None
                data_list[idx]['left_hand_bbox'] = np.empty((0, 5))
                data_list[idx]['left_hand_valid'] = False
                data_list[idx]['right_hand_bbox'] = np.empty((0, 5))
                data_list[idx]['right_hand_valid'] = False  
                
                if dist_left_arm[left_id] < dist_right_arm[right_id]:
                    assign_hand = 'left_hand'
                    assign_id = left_id
                    
                elif dist_left_arm[left_id] > dist_right_arm[right_id]:
                    assign_hand = 'right_hand'
                    assign_id = right_id
            
                if assign_hand is not None:
                    data_list[idx][assign_hand+"_bbox"] = hand_bboxes[assign_id]
                    data_list[idx][assign_hand+"_valid"] = True
    return data_list


def init_hand_detector(args):
    """
        Init hand detect model using detectron.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.hand_det_config)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.hand_det_checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.bbox_thr
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.hand_nms_thr
    hand_det_model = DefaultPredictor(cfg)
    return hand_det_model


def inference_hand_model(hand_det_model, frame, wholebody_kps2d_results, args):
    hand_res = hand_det_model(frame)
    hand_bboxes_coords = hand_res['instances'].pred_boxes.tensor
    hand_bboxes_scores = hand_res['instances'].scores
    
    if hand_bboxes_coords.shape[0] == 0:
            hand_bboxes = torch.empty((0, 5), 
                dtype=hand_bboxes_coords.dtype, device=hand_bboxes_coords.device)
    else:
        hand_bboxes = torch.cat(
            [hand_bboxes_coords, hand_bboxes_scores.unsqueeze(-1)],  dim=-1)
        
    hand_bboxes = hand_bboxes.cpu().numpy()
    
    hand_bbox_wholebody_kps2d_results = assign_handbbox_for_eachperson(args.kpt_thr, wholebody_kps2d_results, hand_bboxes) 
    return hand_bbox_wholebody_kps2d_results


def get_area(bbox, kpts):
    if len(bbox) != 0:
        x1, y1, x2, y2, _ = bbox
    else:
        x1 = np.min(kpts[:, 0])
        y1 = np.min(kpts[:, 1])
        x2 = np.max(kpts[:, 0])
        y2 = np.max(kpts[:, 1])
    return (x2 - x1) * (y2 - y1)


def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--person-det-config', default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py')
    parser.add_argument('--person-det-checkpoint', default='workspace/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
    parser.add_argument('--person-det-cat-id', default=1)
    
    parser.add_argument('--hand-det-config', default='workspace/configs/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml')
    parser.add_argument('--hand-det-checkpoint', default='workspace/checkpoints/model_0529999.pth')
    parser.add_argument('--hand-nms-thr', default=0.5)
    
    # person detection and hand detection share same bbox threshold
    parser.add_argument('--bbox-thr', default=0.9)
    
    # 256 x 192
    parser.add_argument('--wholebody-kps2d-config', default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_coco_wholebody_256x192.py')
    parser.add_argument('--wholebody-kps2d-checkpoint', default='workspace/checkpoints/tcformer_coco-wholebody_256x192-a0720efa_20220627.pth')
       
    # 256 x 256
    parser.add_argument('--hand-kps2d-config', default='configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/ca_res50_coco_wholebody_hand_256x256.py')
    # parser.add_argument('--hand-kps2d-checkpoint', default='workspace/checkpoints/res50_coco_wholebody_hand_256x256-8dbc750c_20210908.pth')
    parser.add_argument('--hand-kps2d-checkpoint', default='work_dirs/ca_res50_coco_wholebody_hand_256x256/best_AUC_epoch_80.pth')
    
    parser.add_argument('--kpt-thr', default=0.3)
    
    parser.add_argument('--video-path', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--device', default='cuda:0')
    
    # Track
    parser.add_argument('--use-oks-tracking', action='store_true')
    parser.add_argument('--tracking-thr', default=0.3, type=float)
    
    # Filter large hand
    parser.add_argument('--filter-large-hand', action='store_true')
    parser.add_argument('--hand-area-thr', default=0.02)
    
    return parser


def main(args):
       
    person_det_model = init_detector(args.person_det_config, args.person_det_checkpoint, device=args.device.lower())
    hand_det_model = init_hand_detector(args)
    wholebody_kps2d_model = init_pose_model(args.wholebody_kps2d_config, args.wholebody_kps2d_checkpoint, device=args.device.lower())
    hand_kps2d_model = init_pose_model(args.hand_kps2d_config, args.hand_kps2d_checkpoint, device=args.device.lower())
    
    video = mmcv.VideoReader(args.video_path)
    video_output_list = []
    
    next_id = 0
    frame_hand_bbox_wholebody_kps2d_results = []
    
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        
        lastframe_hand_bbox_wholebody_kps2d_results = frame_hand_bbox_wholebody_kps2d_results
        
        # detect people
        mmdet_results = inference_detector(person_det_model, cur_frame)
        person_results = process_mmdet_results(mmdet_results, args.person_det_cat_id)
        
        # detect wholebody2d_kps for each person in single image
        wholebody_kps2d_results, _ = inference_top_down_pose_model(
            wholebody_kps2d_model,
            cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=wholebody_kps2d_model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(wholebody_kps2d_model.cfg.data['test']['dataset_info']),
            return_heatmap=False,
            outputs=None,)
        # wholebody_kps2d_results: [dict(bbox, keypoints), ...]
        
        # assign left(right) hand for each person
        hand_bbox_wholebody_kps2d_results = inference_hand_model(
            hand_det_model,
            cur_frame,
            wholebody_kps2d_results,
            args,)
        # hand_bbox_wholebody_kps2d_results: [dict(bbox, keypoints, left_hand_bbox, left_hand_valid, right_hand_bbox, right_hand_valid), ...]
        
        hand_bbox_wholebody_kps2d_results, next_id = get_track_id(
            hand_bbox_wholebody_kps2d_results,
            lastframe_hand_bbox_wholebody_kps2d_results,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)
        
        # fix bug here
        frame_hand_bbox_wholebody_kps2d_results = hand_bbox_wholebody_kps2d_results
        
        hand_results_single_frame = []
        for person in hand_bbox_wholebody_kps2d_results:
            # person: dict(bbox, )
            if person['left_hand_valid']:
                hand_results_single_frame.append({'bbox': person['left_hand_bbox']})
            if person['right_hand_valid']:
                hand_results_single_frame.append({'bbox': person['right_hand_bbox']})
        
        hand_kps2d_results, _ = inference_top_down_pose_model(
            hand_kps2d_model,
            cur_frame,
            hand_results_single_frame,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=hand_kps2d_model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(hand_kps2d_model.cfg.data['test'].get('dataset_info', None)),
            return_heatmap=False,
            outputs=None,) 
        # hand_kps2d_results: [dict(bbox, keypoints), ...]
        
        hand_kps2d_hand_bbox_wholebody_2d_results = copy.deepcopy(hand_bbox_wholebody_kps2d_results)
        hand_idx = 0
        for person in hand_kps2d_hand_bbox_wholebody_2d_results:
            person['person_bbox'] = person['bbox']
            del person['bbox']
            person['wholebody_keypoints'] = person['keypoints']
            del person['keypoints']
            
            if person['left_hand_valid']:
                person['left_hand_keypoints'] = hand_kps2d_results[hand_idx]['keypoints']
                hand_idx += 1
            else:
                person['left_hand_keypoints'] = np.empty((0, 3))
                
            if person['right_hand_valid']:
                person['right_hand_keypoints'] = hand_kps2d_results[hand_idx]['keypoints']
                hand_idx += 1
            else:
                person['right_hand_keypoints'] = np.empty((0, 3))
        
             
        single_frame_dict = dict()
        for person in hand_kps2d_hand_bbox_wholebody_2d_results:
            # todo filter large hand bbox
            if args.filter_large_hand:
                
                person_area = get_area(person['person_bbox'], person['wholebody_keypoints'])
                left_hand_area = get_area(person['left_hand_bbox'], person['wholebody_keypoints'][-42:-21, :])
                right_hand_area = get_area(person['right_hand_bbox'], person['wholebody_keypoints'][-21:, :])
                
                if left_hand_area > person_area * args.hand_area_thr:
                    person['left_hand_valid'] = False
                    person['left_hand_bbox'] = np.empty((0, 5))
                    person['left_hand_keypoints'] = np.empty((0, 3))
                    
                if right_hand_area > person_area * args.hand_area_thr:
                    person['right_hand_valid'] = False
                    person['right_hand_bbox'] = np.empty((0, 5))
                    person['right_hand_keypoints'] = np.empty((0, 3))
            
            single_frame_dict[person['track_id']] = person
                
        # video_output_list.append(hand_kps2d_hand_bbox_wholebody_2d_results)
        video_output_list.append(single_frame_dict)

    name = (os.path.basename(args.video_path)).split('.')[0]
    out_file_name = f'video_{name}_kpts2d.pkl'
    with open(os.path.join(args.out_root, out_file_name), 'wb') as fin:
        pickle.dump(video_output_list, fin)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)