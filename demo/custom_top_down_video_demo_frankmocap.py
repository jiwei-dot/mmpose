# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser


import torch
import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result, get_track_id)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    
    parser.add_argument(
        '--hand_det_config', 
        default='workspace/configs/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml',
        type=str,
        help='Config file for hand detection')
    
    parser.add_argument(
        '--hand_det_checkpoint', 
        default='workspace/checkpoints/model_0529999.pth',
        type=str,
        help='Checkpoint file for hand detection')
    
    parser.add_argument(
        '--hand_pose_config', 
        default="configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/res50_coco_wholebody_hand_256x256.py",
        type=str,
        help='Config file for hand pose')
    
    parser.add_argument(
        '--hand_pose_checkpoint', 
        default='workspace/checkpoints/res50_coco_wholebody_hand_256x256-8dbc750c_20210908.pth',
        type=str,
        help='Checkpoint file for hand pose')
    
    parser.add_argument(
        '--person_det_config', 
        default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        type=str,
        help='Config file for person detection')
    
    parser.add_argument(
        '--person_det_checkpoint', 
        default='workspace/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        type=str,
        help='Checkpoint file for person detection')
    
    parser.add_argument(
        '--person_pose_config', 
        default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_coco_wholebody_256x192.py',
        type=str,
        help='Config file for person pose')
    
    parser.add_argument(
        '--person_pose_checkpoint', 
        default='workspace/checkpoints/tcformer_coco-wholebody_256x192-a0720efa_20220627.pth',
        type=str,
        help='Checkpoint file for person pose')
    
    parser.add_argument(
        '--video-path', 
        default='workspace/videos/hw-094-hd.mp4',
        type=str, 
        help='Video path')
    
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    
    parser.add_argument(
        '--out-video-root',
        default='workspace',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model, only used in person detection')
    
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='Bounding box score threshold')
    
    parser.add_argument(
        '--kpt-thr', 
        type=float, 
        default=-1, 
        help='Keypoint score threshold')
    
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.5,
        help='Nms threshold')
    
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')


    args = parser.parse_args()
    assert args.show or (args.out_video_root != '')
    
    
    print("Creating person detector...")
    person_detect_model = init_detector(
        args.person_det_config, args.person_det_checkpoint, device=args.device.lower())
    
    
    print("Creating hand detector...")
    cfg = get_cfg()
    cfg.merge_from_file(args.hand_det_config)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.hand_det_checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.bbox_thr
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thr
    hand_detect_model = DefaultPredictor(cfg)
    
    
    print("Creating person keypoint_2d estimator...")
    person_keypoints2d_model = init_pose_model(
        args.person_pose_config, args.person_pose_checkpoint, device=args.device.lower())
    
    
    print("Creating hand keypoint_2d estimator...")
    hand_keypoints2d_model = init_pose_model(
        args.hand_pose_config, args.hand_pose_checkpoint, device=args.device.lower())
    
    person_dataset = person_keypoints2d_model.cfg.data['test']['type']
    hand_dataset = hand_keypoints2d_model.cfg.data['test']['type']
    
    
    # get datasetinfo
    person_dataset_info = person_keypoints2d_model.cfg.data['test'].get('dataset_info', None)
    person_dataset_info = DatasetInfo(person_dataset_info)
    
    hand_dataset_info = hand_keypoints2d_model.cfg.data['test'].get('dataset_info', None)
    hand_dataset_info = DatasetInfo(hand_dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'


    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True


    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root, f'vis_{os.path.basename(args.video_path)}'), 
            fourcc, fps, size)


    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in hand_keypoints2d_model.cfg.data.test.data_cfg
        indices = hand_keypoints2d_model.cfg.data.test.data_cfg['frame_indices_test']


    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None


    # 按照`frankmocap`的思路,先检测人以及每个人的关键点,然后在一张图片上检测所有手,
    # 为每个人按照一定规则分配左手和右手(可能因为没检测出来不分配)
    
    
    # only consider single person

    next_id = 0
    person_keypoints2d_results = []
    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        
        person_keypoints2d_results_last = person_keypoints2d_results
        
        ### 1. person detection
        # ----------------------------------------------------------------------
        
        mmdet_results = inference_detector(person_detect_model, cur_frame)
        # mmdet_results: [N1x5, N2x5, ...]
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        # person_results: 当检测到物体时为[{bbox: shape[5, ]}, {bbox: shape[5, ]}, ...]， 否则为空list

       
        if len(person_results) == 0:
            # 没检测到物体, do something
            # todo
            pass
        # ----------------------------------------------------------------------
 
        if args.use_multi_frames:
            frames = collect_multi_frames(
                video, frame_id, indices, args.online)
            
            
        single_person = True
        if single_person:
            person_results = [person_results[0], ]
        
        
        ### 2. for every person, do wholebody_2d keypoints detection
        # ---------------------------------------------------------------------------------

        person_keypoints2d_results, _ = inference_top_down_pose_model(
            person_keypoints2d_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=person_dataset,
            dataset_info=person_dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        # person_keypoints2d_results: [{bbox: shape[5, ], keypoints: shape[133, 3]}, ...], 也可能由于未检测到物体为空list
        
        person_keypoints2d_results, next_id = get_track_id(
            person_keypoints2d_results,
            person_keypoints2d_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)
        # person_keypoints2d_results: [{bbox: shape[5, ], keypoints: shape[133, 3], area, track_id}, ...], 也可能由于未检测到物体为空list
        # ---------------------------------------------------------------------------------
        
        
        ### 3. 检测hand, 既可以运行一个单独的hand_detector, 也可以利用wholebody_keypoints的结果预估
        ###    为检测到的每个人分配左右手
        # ---------------------------------------------------------------------------------
        
        fast_mode = False
        if fast_mode:
            # 使用wholebody_keypoints的结果预估
            # todo
            pass
        else:
            
            # ------------------------ 单张图片检测hand -------------------------------
            res = hand_detect_model(cur_frame)
            # res: dict
            hand_bboxes_coords = res['instances'].pred_boxes.tensor
            hand_bboxes_scores = res['instances'].scores
            if hand_bboxes_coords.shape[0] == 0:
                hand_bboxes = torch.empty((0, 5), 
                    dtype=hand_bboxes_coords.dtype, device=hand_bboxes_coords.device)
            else:
                hand_bboxes = torch.cat(
                    [hand_bboxes_coords, hand_bboxes_scores.unsqueeze(-1)],  dim=-1)
            hand_bboxes = hand_bboxes.cpu().numpy()
            # ------------------------ 单张图片检测hand -------------------------------
            
            
            num_hand_bboxes = len(hand_bboxes)
            if num_hand_bboxes == 0:
                for idx, person in enumerate(person_keypoints2d_results):
                    person_keypoints2d_results[idx]['left_hand_bbox'] = np.empty((0, 5))
                    person_keypoints2d_results[idx]['left_hand_valid'] = False
                    person_keypoints2d_results[idx]['right_hand_bbox'] = np.empty((0, 5))
                    person_keypoints2d_results[idx]['right_hand_valid'] = False
            else:
                for idx, person in enumerate(person_keypoints2d_results):  
                    
                    dist_left_arm = np.ones((num_hand_bboxes, )) * float('inf')
                    dist_right_arm = np.ones((num_hand_bboxes, )) * float('inf')
                    person_keypoints2d = person['keypoints']
                    
                    # left arm
                    if person_keypoints2d[9][-1] > args.kpt_thr and person_keypoints2d[7][-1] > args.kpt_thr:
                        dist_wrist_elbow = np.linalg.norm(person_keypoints2d[9][:2] - person_keypoints2d[7][:2])
                        c_x = (hand_bboxes[:, 0] + hand_bboxes[:, 2]) / 2
                        c_y = (hand_bboxes[:, 1] + hand_bboxes[:, 3]) / 2
                        center = np.stack((c_x, c_y), axis=-1)
                        dist_bbox_ankle = np.linalg.norm(center - person_keypoints2d[9][:2], axis=-1)
                        mask = dist_bbox_ankle < dist_wrist_elbow * 1.5
                        dist_left_arm[mask] = dist_bbox_ankle[mask]

                    # right arm
                    if person_keypoints2d[10][-1] > args.kpt_thr and person_keypoints2d[8][-1] > args.kpt_thr:
                        dist_wrist_elbow = np.linalg.norm(person_keypoints2d[10][:2] - person_keypoints2d[8][:2])
                        c_x = (hand_bboxes[:, 0] + hand_bboxes[:, 2]) / 2
                        c_y = (hand_bboxes[:, 1] + hand_bboxes[:, 3]) / 2
                        center = np.stack((c_x, c_y), axis=-1)
                        dist_bbox_ankle = np.linalg.norm(center - person_keypoints2d[10][:2], axis=-1)
                        mask = dist_bbox_ankle < dist_wrist_elbow * 1.5
                        dist_right_arm[mask] = dist_bbox_ankle[mask]
                        
                    left_id = np.argmin(dist_left_arm)
                    right_id = np.argmin(dist_right_arm)
                    
                    if dist_left_arm[left_id] < float('inf'):
                        person_keypoints2d_results[idx]['left_hand_bbox'] = hand_bboxes[left_id]
                        person_keypoints2d_results[idx]['left_hand_valid'] = True
                    else:
                        person_keypoints2d_results[idx]['left_hand_bbox'] = np.empty((0, 5))
                        person_keypoints2d_results[idx]['left_hand_valid'] = False
                        
                    if dist_right_arm[right_id] < float('inf'):
                        person_keypoints2d_results[idx]['right_hand_bbox'] = hand_bboxes[right_id]
                        person_keypoints2d_results[idx]['right_hand_valid'] = True
                    else:
                        person_keypoints2d_results[idx]['right_hand_bbox'] = np.empty((0, 5))
                        person_keypoints2d_results[idx]['right_hand_valid'] = False
                        
        # person_keypoints2d_results: [{bbox, keypoints, left_hand_box, left_hand_valid, right_hand_box, area, track_id}, ...]
        # left_hand_box, right_hand_box大小可能为[0x5]
        # ---------------------------------------------------------------------------------

        
        ### 4. 对每个人的每个hand, 运行2d hand keypoints模型
        # ---------------------------------------------------------------------------------
        hand_results = []
        for person in person_keypoints2d_results:
            if person['left_hand_valid']:
                hand_results.append({'bbox': person['left_hand_bbox']})
            if person['right_hand_valid']:
                hand_results.append({'bbox': person['right_hand_bbox']})
        
        hand_keypoints2d_results, _ = inference_top_down_pose_model(
            hand_keypoints2d_model,
            frames if args.use_multi_frames else cur_frame,
            hand_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=hand_dataset,
            dataset_info=hand_dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        # hand_keypoints2d_results: [{bbox, keypoints}, ...], 也可能为空list
        
        
        # 针对检测到的每个人,他的左右手可能没有被检测出来,使用单独的hand_keypoints2d_model检测不出来关键点
        # 如果手被检测出来，使用hand_keypoints2d_model预测的结果(要不要和tcformer预测结果进行对比)
        # 如果手没被检测出来，使用上一帧预测的结果, 直接用tcformer的结果可能会有问题

        tmp_idx = 0
        for person in person_keypoints2d_results:
            # person: dict(bbox, keypoints, area, track_id, left_hand_bbox, left_hand_valid, right_hand_bbox, right_hand_valid)
            if person['left_hand_valid']:
                person['left_hand_keypoints'] = hand_keypoints2d_results[tmp_idx]['keypoints']
                tmp_idx += 1
            else:
                tmp_person = None
                for person_last in person_keypoints2d_results_last:
                    if person_last['track_id'] == person['track_id']:
                        tmp_person = person_last
                        break
                if tmp_person is None:
                    # 没检测到hand, 同时上一帧也没有对应的信息
                    person['left_hand_keypoints'] = torch.empty((0, 3), device=args.device)
                else:
                    # 没检测到hand, 但上一帧有对应的信息
                    person['left_hand_keypoints'] = tmp_person['left_hand_keypoints']
                    person['left_hand_bbox'] = tmp_person['left_hand_bbox']
                    person['left_hand_valid'] = True
                  
            if person['right_hand_valid']:
                person['right_hand_keypoints'] = hand_keypoints2d_results[tmp_idx]['keypoints']
                tmp_idx += 1
            else:
                tmp_person = None
                for person_last in person_keypoints2d_results_last:
                    if person_last['track_id'] == person['track_id']:
                        tmp_person = person_last
                        break
                if tmp_person is None:
                    person['right_hand_keypoints'] = torch.empty((0, 3), device=args.device)
                else:
                    person['right_hand_keypoints'] = tmp_person['right_hand_keypoints']
                    person['right_hand_bbox'] = tmp_person['right_hand_bbox']
                    person['right_hand_valid'] = True
            
            if person['left_hand_valid']:
                person['keypoints'][91: 112] = person['left_hand_keypoints'] 
            
            if person['right_hand_valid']:
                person['keypoints'][112: 133] = person['right_hand_keypoints']

        vis_frame = vis_pose_result(
            person_keypoints2d_model,
            cur_frame,
            person_keypoints2d_results,
            dataset=person_dataset,
            dataset_info=person_dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)
        
        cv2.imwrite(f'workspace/images/{frame_id}.jpg', vis_frame)
        videoWriter.write(vis_frame)
        # ---------------------------------------------------------------------------------
                    
    videoWriter.release()        
                    

if __name__ == '__main__':
    main()
