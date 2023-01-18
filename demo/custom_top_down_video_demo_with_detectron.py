# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
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
        default='workspace/videos/hw-086-hd.mp4',
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
        default=0.1, 
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

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    
    
    print("Creating hand detector...")
    cfg = get_cfg()
    cfg.merge_from_file(args.hand_det_config)
    cfg.MODEL.WEIGHTS = args.hand_det_checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.bbox_thr
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thr
    hand_detect_model = DefaultPredictor(cfg)
    
    
    print("Creating hand keypoint_2d estimator...")
    hand_keypoints2d_model = init_pose_model(
        args.hand_pose_config, args.hand_pose_checkpoint, device=args.device.lower())
    
    
    print("Creating person detector...")
    person_detect_model = init_detector(
        args.person_det_config, args.person_det_checkpoint, device=args.device.lower())
    
    
    print("Creating person keypoint_2d estimator...")
    person_keypoints2d_model = init_pose_model(
        args.person_pose_config, args.person_pose_checkpoint, device=args.device.lower())


    hand_dataset = hand_keypoints2d_model.cfg.data['test']['type']
    person_dataset = person_keypoints2d_model.cfg.data['test']['type']
    
    
    # get datasetinfo
    hand_dataset_info = hand_keypoints2d_model.cfg.data['test'].get('dataset_info', None)
    hand_dataset_info = DatasetInfo(hand_dataset_info)
    
    person_dataset_info = person_keypoints2d_model.cfg.data['test'].get('dataset_info', None)
    person_dataset_info = DatasetInfo(person_dataset_info)


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


    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        
        # ---------------------------------------------------------------------------------
        bbox_tensor = hand_detect_model(cur_frame)['instances'].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        
        bboxes = np.column_stack((bboxes, np.ones(bboxes.shape[0])))
        hand_results = []
        
        for box in bboxes:
            tmp_dict = dict()
            tmp_dict['bbox'] = box
            hand_results.append(tmp_dict)
        # ---------------------------------------------------------------------------------
        
        
        # ---------------------------------------------------------------------------------
        mmdet_results = inference_detector(person_detect_model, cur_frame)
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        # ---------------------------------------------------------------------------------

        if args.use_multi_frames:
            frames = collect_multi_frames(
                video, frame_id, indices, args.online)

        # test a single image, with a list of bboxes.
        
        # hand
        hand_keypoints2d_results, hand_returned_outputs = inference_top_down_pose_model(
            hand_keypoints2d_model,
            frames if args.use_multi_frames else cur_frame,
            hand_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=hand_dataset,
            dataset_info=hand_dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        # wholebody
        person_keypoints2d_results, person_returned_outputs = inference_top_down_pose_model(
            person_keypoints2d_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=person_dataset,
            dataset_info=person_dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        vis_frame_hand = vis_pose_result(
            hand_keypoints2d_model,
            cur_frame,
            hand_keypoints2d_results,
            dataset=hand_dataset,
            dataset_info=hand_dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)
        
        vis_frame_person = vis_pose_result(
            person_keypoints2d_model,
            cur_frame,
            person_keypoints2d_results,
            dataset=person_dataset,
            dataset_info=person_dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)
        
        H, W, C = vis_frame_hand.shape
        vis_frame = np.zeros((H, 2 * W, C), dtype=vis_frame_hand.dtype)
        
        vis_frame[:, :W, :] = vis_frame_hand
        vis_frame[:, W:, :] = vis_frame_person

        if args.show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
