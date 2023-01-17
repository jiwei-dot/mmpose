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
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', 
        type=float, 
        default=0.1, 
        help='Keypoint score threshold')
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
    cfg.merge_from_file(args.det_config)
    cfg.MODEL.WEIGHTS = args.det_checkpoint
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.bbox_thr
    hand_detect_model = DefaultPredictor(cfg)
    
    print("Creating 2d keypoint estimator...")
    keypoints2d_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    # HandCocoWholeBodyDataset
    dataset = keypoints2d_model.cfg.data['test']['type']

    # get datasetinfo
    dataset_info = keypoints2d_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

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
        assert 'frame_indices_test' in keypoints2d_model.cfg.data.test.data_cfg
        indices = keypoints2d_model.cfg.data.test.data_cfg['frame_indices_test']

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        
        bbox_tensor = hand_detect_model(cur_frame)['instances'].pred_boxes
        bboxes = bbox_tensor.tensor.cpu().numpy()
        
        bboxes = np.column_stack((bboxes, np.ones(bboxes.shape[0])))
        hand_results = []
        
        for box in bboxes:
            tmp_dict = dict()
            tmp_dict['bbox'] = box
            hand_results.append(tmp_dict)

        if args.use_multi_frames:
            frames = collect_multi_frames(
                video, frame_id, indices, args.online)

        print("len(hand_results)", len(hand_results))
        # test a single image, with a list of bboxes.
        keypoints2d_results, returned_outputs = inference_top_down_pose_model(
            keypoints2d_model,
            frames if args.use_multi_frames else cur_frame,
            hand_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        vis_frame = vis_pose_result(
            keypoints2d_model,
            cur_frame,
            keypoints2d_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)

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
