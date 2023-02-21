import warnings
warnings.filterwarnings("ignore")
import copy
import os
import os.path as osp
import cv2
import numpy as np
from argparse import ArgumentParser
import torch


import mmcv
from mmcv import Config
from mmpose.apis import (init_pose_model, process_mmdet_results, collect_multi_frames,
                        inference_top_down_pose_model, get_track_id, extract_pose_sequence,
                        inference_pose_lifter_model, vis_3d_pose_result)
from mmpose.models import TopDown, PoseLifter
from mmpose.datasets import DatasetInfo
from mmpose.core import Smoother


try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
 

from mmpose.apis import vis_pose_result
from mmpose.datasets.pipelines.custom_hand_postprocessing import *
    
    
def coco_to_body(keypoints):
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


def coco_to_lhand(keypoints):
    indices = [9, ]
    indices.extend(list(range(91, 112)))
    return keypoints[indices, ...]


def coco_to_rhand(keypoints):
    indices = [10, ]
    indices.extend(list(range(112, 133)))
    return keypoints[indices, ...]


def coco_to_lfoot(keypoints):
    return keypoints[[15, 17, 18, 19], ...]


def coco_to_rfoot(keypoints):
    return keypoints[[16, 20, 21, 22], ...]


def get_parser():
    
    parser = ArgumentParser()
    
    
    ## --------------------- config and checkpoint file setting ---------------------------
    
    # for person detection
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--person_det_config', 
        type=str,
        default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        help='Config file for person detection')
    
    parser.add_argument(
        '--person_det_checkpoint', 
        type=str,
        default='workspace/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        help='Checkpoint file for person detection')
    
    
    # for hand detection
    # --------------------------------------------------------------------------------------
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
    
    
    # for wholebody 2d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--wholebody2d_pose_config',
        type=str,
        default='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/tcformer_coco_wholebody_256x192.py',
        help='Config file for the wholebody 2D pose detector')
    parser.add_argument(
        '--wholebody2d_pose_checkpoint',
        type=str,
        default='workspace/checkpoints/tcformer_coco-wholebody_256x192-a0720efa_20220627.pth',
        help='Checkpoint file for the wholebody 2D pose detector')
    
    
    # for hand 2d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--hand2d_pose_config', 
        default="configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/res50_coco_wholebody_hand_256x256.py",
        type=str,
        help='Config file for hand 2D pose detector')
    
    parser.add_argument(
        '--hand2d_pose_checkpoint', 
        default='workspace/checkpoints/res50_coco_wholebody_hand_256x256-8dbc750c_20210908.pth',
        type=str,
        help='Checkpoint file for hand 2D pose detector')
    
    
    # for body 3d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--body3d_lifter_config',
        type=str,
        default='configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py',
        help='Config file for the 2nd stage body lifter model')
    
    parser.add_argument(
        '--body3d_lifter_checkpoint',
        type=str,
        default='workspace/checkpoints/simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
        help='Checkpoint file for the 2nd stage body lifter model')
    
    
    # 之后的工作: 左和右用同一个模型
    # for left_hand 3d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--lefthand3d_lifter_config',
        type=str,
        default='configs/hand/3d_kpt_sview_rgb_img/simplebaseline3d/simplebaseline3d_h3wb_left_hand.py',
        help='Config file for the 2nd stage left_hand lifter model')
    
    parser.add_argument(
        '--lefthand3d_lifter_checkpoint',
        type=str,
        default='work_dirs/simplebaseline3d_h3wb_left_hand/best_MPJPE_epoch_200.pth',
        help='Checkpoint file for the 2nd stage left_hand lifter model')
    
    
    # for right_hand 3d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--righthand3d_lifter_config',
        type=str,
        default='configs/hand/3d_kpt_sview_rgb_img/simplebaseline3d/simplebaseline3d_h3wb_right_hand.py',
        help='Config file for the 2nd stage right_hand lifter model')
    
    parser.add_argument(
        '--righthand3d_lifter_checkpoint',
        type=str,
        default='work_dirs/simplebaseline3d_h3wb_right_hand/best_MPJPE_epoch_200.pth',
        help='Checkpoint file for the 2nd stage right_hand lifter model')
    
    
    # for left_foot 3d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--leftfoot3d_lifter_config',
        type=str,
        default='configs/foot/3d_kpt_sview_rgb_img/pose_lift/h3wb/simplebaseline3d_h3wb_left_foot.py',
        help='Config file for the 2nd stage left_foot lifter model')
    
    parser.add_argument(
        '--leftfoot3d_lifter_checkpoint',
        type=str,
        default='work_dirs/simplebaseline3d_h3wb_left_foot/best_MPJPE_epoch_160.pth',
        help='Checkpoint file for the 2nd stage left_foot lifter model')
    
    
    # for right_foot 3d keypoints
    # --------------------------------------------------------------------------------------
    parser.add_argument(
        '--rightfoot3d_lifter_config',
        type=str,
        default='configs/foot/3d_kpt_sview_rgb_img/pose_lift/h3wb/simplebaseline3d_h3wb_right_foot.py',
        help='Config file for the 2nd stage right_foot lifter model')
    
    parser.add_argument(
        '--rightfoot3d_lifter_checkpoint',
        type=str,
        default='work_dirs/simplebaseline3d_h3wb_right_foot/best_MPJPE_epoch_120.pth',
        help='Checkpoint file for the 2nd stage right_foot lifter model')
    
    ## --------------------- config and checkpoint file setting ---------------------------
    
      
    parser.add_argument('--video-path', type=str, default='', help='Video path')
    
    
    # 针对某一视频，使用已经检测好的wholebody2d文件
    parser.add_argument('--use-pkl', action='store_true')
    parser.add_argument('--pkl-file', type=str, default='')
    
    
    # 每帧只处理一个人
    parser.add_argument('--single-person', action='store_false')
    
    
    # 针对手部获取包围框，True时使用wholebody2d的检测结果，否则使用单独一个手的检测模型
    parser.add_argument('--fast-mode', action='store_true')
    
    
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')

    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    
    parser.add_argument(
        '--num-instances',
        type=int,
        default=-1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')

    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')

    parser.add_argument(
        '--out-video-root',
        type=str,
        default='vis_results',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')

    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.9,
        help='Bounding box score threshold')
    
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.5,
        help='Nms threshold')

    parser.add_argument('--kpt-thr', type=float, default=0)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the 2D pose estimation '
        'results. See also --smooth-filter-cfg.')
    
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')
    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the 2D pose'
        'detection stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')
    
    return parser


def get_wholebody2d(args, video):
    pose_det_results_list = []
    
    person_det_model = init_detector(
        args.person_det_config, 
        args.person_det_checkpoint, 
        device=args.device.lower())
    
    wholebody2d_pose_model = init_pose_model(
        args.wholebody2d_pose_config, 
        args.wholebody2d_pose_checkpoint, 
        device=args.device.lower())
    
    hand_det_model = None
    hand2d_pose_model = None
    
    op_assignhandbbox = AssignHandBBoxForEachPerson(kpt_thr=args.kpt_thr)
    op_assignhandkeypoints2d = AssignHandKeypoints2dForEachPerson()
    op_filterlargehand = FilterLargeHandForEachPerson()
    
    next_id = 0
    person_keypoints2d_results = []
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)): 
 
        person_keypoints2d_results_last = person_keypoints2d_results
        mmdet_results = inference_detector(person_det_model, cur_frame)
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        if len(person_results) == 0:
            pose_det_results_list.append([])
            continue
        
        if args.single_person:
            person_results = [person_results[0], ]
        
        person_keypoints2d_results, _ = inference_top_down_pose_model(
            wholebody2d_pose_model,
            cur_frame,                      # 单帧预测
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=wholebody2d_pose_model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(wholebody2d_pose_model.cfg.data['test'].get('dataset_info', None)),
            return_heatmap=False,
            outputs=None)
        
        person_keypoints2d_results, next_id = get_track_id(
            person_keypoints2d_results,
            copy.deepcopy(person_keypoints2d_results_last),
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)
        
        if args.fast_mode:
            raise NotImplementedError
        else:
            if hand_det_model is None:
                from detectron2.config import get_cfg
                from detectron2.engine import DefaultPredictor
                cfg = get_cfg()
                cfg.merge_from_file(args.hand_det_config)
                cfg.MODEL.DEVICE = args.device
                cfg.MODEL.WEIGHTS = args.hand_det_checkpoint
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.bbox_thr
                cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thr
                hand_det_model = DefaultPredictor(cfg)
            
            if hand2d_pose_model is None:
                hand2d_pose_model = init_pose_model(
                    args.hand2d_pose_config, 
                    args.hand2d_pose_checkpoint, 
                    device=args.device.lower())
        
        hand_res = hand_det_model(cur_frame)
        # hand_res: dict
        hand_bboxes_coords = hand_res['instances'].pred_boxes.tensor
        hand_bboxes_scores = hand_res['instances'].scores
        if hand_bboxes_coords.shape[0] == 0:
            hand_bboxes = torch.empty((0, 5), 
                dtype=hand_bboxes_coords.dtype, device=hand_bboxes_coords.device)
        else:
            hand_bboxes = torch.cat(
                [hand_bboxes_coords, hand_bboxes_scores.unsqueeze(-1)],  dim=-1)
        hand_bboxes = hand_bboxes.cpu().numpy()
        
        person_keypoints2d_results = op_assignhandbbox(person_keypoints2d_results, hand_bboxes)
    
        hand_results = []
        for person in person_keypoints2d_results:
            if person['left_hand_valid']:
                hand_results.append({'bbox': person['left_hand_bbox']})
            if person['right_hand_valid']:
                hand_results.append({'bbox': person['right_hand_bbox']})
        
        hand_keypoints2d_results, _ = inference_top_down_pose_model(
            hand2d_pose_model,
            cur_frame,
            hand_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=hand2d_pose_model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(hand2d_pose_model.cfg.data['test'].get('dataset_info', None)),
            return_heatmap=False,
            outputs=None)

        person_keypoints2d_results  = op_assignhandkeypoints2d(person_keypoints2d_results, person_keypoints2d_results_last, hand_keypoints2d_results)
        person_keypoints2d_results = op_filterlargehand(person_keypoints2d_results)       
        pose_det_results_list.append(person_keypoints2d_results)
        
        vis_frame = vis_pose_result(
            wholebody2d_pose_model,
            cur_frame,
            person_keypoints2d_results,
            dataset=wholebody2d_pose_model.cfg.data['test']['type'],
            dataset_info=DatasetInfo(wholebody2d_pose_model.cfg.data['test'].get('dataset_info', None)),
            kpt_score_thr=0,
            radius=1,
            thickness=args.thickness,
            show=False)

        cv2.imwrite(f'workspace/images/{frame_id}.jpg', vis_frame)

    return pose_det_results_list
    


def main():
    
    parser = get_parser()
    args = parser.parse_args()
    
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Failed to load video file {args.video_path}'
    
    print('Stage 1: 2D pose detection.')
    # -------------------------------------------------------------------------
    if args.use_pkl:
        assert args.pkl_file, 'please provide pre-detected pkl file'
        import pickle
        with open(args.pkl_file, 'rb') as fin:
            pose_det_results_list = pickle.load(fin)
    else:
        pose_det_results_list = get_wholebody2d(args, video)


        # import pickle
        # with open('res.pkl', 'wb') as fin:
        #     pickle.dump(pose_det_results_list, fin)


    print('Stage 2: 2D-to-3D pose lifting.')
    # ---------------------------------------------------------------------------
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
    
    assert isinstance(body_lift_model, PoseLifter)
    assert isinstance(lhand_lift_model, PoseLifter)
    assert isinstance(rhand_lift_model, PoseLifter)
    assert isinstance(lfoot_lift_model, PoseLifter)
    assert isinstance(rfoot_lift_model, PoseLifter)
    
    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None
        
    # convert keypoint definition
    body_det_results_list = copy.deepcopy(pose_det_results_list)
    lhand_det_results_list = copy.deepcopy(pose_det_results_list)
    rhand_det_results_list = copy.deepcopy(pose_det_results_list)
    lfoot_det_results_list = copy.deepcopy(pose_det_results_list)
    rfoot_det_results_list = copy.deepcopy(pose_det_results_list)

    for body_det_results in body_det_results_list:
        for res in body_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = coco_to_body(keypoints)

    for lhand_det_results in lhand_det_results_list:
        for res in lhand_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = coco_to_lhand(keypoints)    
            
    for rhand_det_results in rhand_det_results_list:
        for res in rhand_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = coco_to_rhand(keypoints)
               
    for lfoot_det_results in lfoot_det_results_list:
        for res in lfoot_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = coco_to_lfoot(keypoints)   
            x1 = np.min(res['keypoints'][:, 0])
            y1 = np.min(res['keypoints'][:, 1])
            x2 = np.max(res['keypoints'][:, 0])
            y2 = np.max(res['keypoints'][:, 1])
            res['bbox'] = np.array([x1, y1, x2, y2])
            res['area'] = (x2 - x1) * (y2 - y1)

    for rfoot_det_results in rfoot_det_results_list:
        for res in rfoot_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = coco_to_rfoot(keypoints)
            x1 = np.min(res['keypoints'][:, 0])
            y1 = np.min(res['keypoints'][:, 1])
            x2 = np.max(res['keypoints'][:, 0])
            y2 = np.max(res['keypoints'][:, 1])
            res['bbox'] = np.array([x1, y1, x2, y2])
            res['area'] = (x2 - x1) * (y2 - y1)
            
    # load temporal padding config from model.data_cfg
    if hasattr(body_lift_model.cfg, 'test_data_cfg'):
        body_data_cfg = body_lift_model.cfg.test_data_cfg
    else:
        body_data_cfg = body_lift_model.cfg.data_cfg
        
    if hasattr(lhand_lift_model.cfg, 'test_data_cfg'):
        lhand_data_cfg = lhand_lift_model.cfg.test_data_cfg
    else:
        lhand_data_cfg = lhand_lift_model.cfg.data_cfg
        
    if hasattr(rhand_lift_model.cfg, 'test_data_cfg'):
        rhand_data_cfg = rhand_lift_model.cfg.test_data_cfg
    else:
        rhand_data_cfg = rhand_lift_model.cfg.data_cfg
        
    if hasattr(lfoot_lift_model.cfg, 'test_data_cfg'):
        lfoot_data_cfg = lfoot_lift_model.cfg.test_data_cfg
    else:
        lfoot_data_cfg = lfoot_lift_model.cfg.data_cfg
        
    if hasattr(rfoot_lift_model.cfg, 'test_data_cfg'):
        rfoot_data_cfg = rfoot_lift_model.cfg.test_data_cfg
    else:
        rfoot_data_cfg = rfoot_lift_model.cfg.data_cfg
    
    # build pose smoother for temporal refinement
    if args.smooth:
        body_smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2)   

        lhand_smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2)  

        rhand_smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2) 

        lfoot_smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2)  

        rfoot_smoother = Smoother(
            filter_cfg=args.smooth_filter_cfg,
            keypoint_key='keypoints',
            keypoint_dim=2) 
    else:
         body_smoother = None
         lhand_smoother = None
         rhand_smoother = None
         lfoot_smoother = None
         rfoot_smoother = None
    
    num_instances = args.num_instances
    
    body_lift_dataset_info = body_lift_model.cfg.data['test'].get(
        'dataset_info', None)
    if body_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        body_lift_dataset_info = DatasetInfo(body_lift_dataset_info)
        
    print('Running 2D-to-3D pose lifting inference...')
    h3wb_wo_face_dataset_info = 'configs/_base_/datasets/h36m_wo_face.py'
    cfg = Config.fromfile(h3wb_wo_face_dataset_info)
    h3wb_wo_face_dataset_info = DatasetInfo(cfg.dataset_info)
    
    for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
    
        body_results_2d = extract_pose_sequence(
            body_det_results_list,
            frame_idx=i,
            causal=body_data_cfg.causal,
            seq_len=body_data_cfg.seq_len,
            step=body_data_cfg.seq_frame_interval)
        
        lhand_results_2d = extract_pose_sequence(
            lhand_det_results_list,
            frame_idx=i,
            causal=lhand_data_cfg.causal,
            seq_len=lhand_data_cfg.seq_len,
            step=lhand_data_cfg.seq_frame_interval)
        
        rhand_results_2d = extract_pose_sequence(
            rhand_det_results_list,
            frame_idx=i,
            causal=rhand_data_cfg.causal,
            seq_len=rhand_data_cfg.seq_len,
            step=rhand_data_cfg.seq_frame_interval)
        
        lfoot_results_2d = extract_pose_sequence(
            lfoot_det_results_list,
            frame_idx=i,
            causal=lfoot_data_cfg.causal,
            seq_len=lfoot_data_cfg.seq_len,
            step=lfoot_data_cfg.seq_frame_interval)
        
        rfoot_results_2d = extract_pose_sequence(
            rfoot_det_results_list,
            frame_idx=i,
            causal=rfoot_data_cfg.causal,
            seq_len=rfoot_data_cfg.seq_len,
            step=rfoot_data_cfg.seq_frame_interval)
        
        if body_smoother:
            body_results_2d = body_smoother.smooth(body_results_2d)
        if lhand_smoother:
            lhand_results_2d = lhand_smoother.smooth(lhand_results_2d)
        if rhand_smoother:
            rhand_results_2d = rhand_smoother.smooth(rhand_results_2d)
        if lfoot_smoother:
            lfoot_results_2d = lfoot_smoother.smooth(lfoot_results_2d)
        if rfoot_smoother:
            rfoot_results_2d = rfoot_smoother.smooth(rfoot_results_2d)
            
        # body 2D -> 3D
        body_lift_results = inference_pose_lifter_model(
            body_lift_model,
            pose_results_2d=body_results_2d,
            dataset='Body3DH36MDataset',
            dataset_info=DatasetInfo(body_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        
        # body_lift_results: [{track_id, keypoints, keypoints_3d}, ]
        
        # left hand 2D -> 3D
        left_hand_lift_results = inference_pose_lifter_model(
            lhand_lift_model,
            pose_results_2d=lhand_results_2d,
            dataset='Hand3DH3WBDataset',
            dataset_info=DatasetInfo(lhand_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        
        # right hand 2D -> 3D
        right_hand_lift_results = inference_pose_lifter_model(
            rhand_lift_model,
            pose_results_2d=rhand_results_2d,
            dataset='Hand3DH3WBDataset',
            dataset_info=DatasetInfo(rhand_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        
        # left foot 2D -> 3D
        left_foot_lift_results = inference_pose_lifter_model(
            lfoot_lift_model,
            pose_results_2d=lfoot_results_2d,
            dataset='Foot3DH3WBDataset',
            dataset_info=DatasetInfo(lfoot_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        
        # right foot 2D -> 3D
        right_foot_lift_results = inference_pose_lifter_model(
            rfoot_lift_model,
            pose_results_2d=rfoot_results_2d,
            dataset='Foot3DH3WBDataset',
            dataset_info=DatasetInfo(rfoot_lift_model.cfg.data['test'].get('dataset_info', None)),
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)
        
        body_hands_and_foots_lift_results = copy.deepcopy(body_lift_results)
        for idx in range(len(body_hands_and_foots_lift_results)):
            tmp_bh, lhand, rhand = body_hands_and_foots_lift_results[idx], left_hand_lift_results[idx], right_hand_lift_results[idx]
            lfoot, rfoot = left_foot_lift_results[idx], right_foot_lift_results[idx]
            
            tmp_bh['keypoints'] = np.concatenate([
                tmp_bh['keypoints'], lhand['keypoints'][:, 1:, ...], rhand['keypoints'][:, 1:, ...],
                lfoot['keypoints'][:, 1:, ...], rfoot['keypoints'][:, 1:, ...]], axis=1)
            
            lhand['keypoints_3d'] += tmp_bh['keypoints_3d'][13, :]
            rhand['keypoints_3d'] += tmp_bh['keypoints_3d'][16, :]
            
            lfoot['keypoints_3d'] += tmp_bh['keypoints_3d'][6, :]
            rfoot['keypoints_3d'] += tmp_bh['keypoints_3d'][3, :]
            
            tmp_bh['keypoints_3d'] = np.concatenate([
                tmp_bh['keypoints_3d'], lhand['keypoints_3d'][1:, ...], rhand['keypoints_3d'][1:, ...],
                lfoot['keypoints_3d'][1:, ...], rfoot['keypoints_3d'][1:, ...]])
            
             
        # Pose processing
        body_and_hands_lift_results_vis = []
        for idx, res in enumerate(body_hands_and_foots_lift_results):
            keypoints_3d = res['keypoints_3d']
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            # keypoints_3d[..., 0] = -keypoints_3d[..., 0] 
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            res['title'] = 'Prediction'
            # res['keypoints'] = pose_det_results[idx]['keypoints']
            res['keypoints'] = res['keypoints'][-1]
            res['bbox'] = pose_det_results[idx]['bbox']
            res['track_id'] = pose_det_results[idx]['track_id'] 
            res['left_hand_bbox'] = pose_det_results[idx]['left_hand_bbox'] 
            res['left_hand_valid'] = pose_det_results[idx]['left_hand_valid']
            res['right_hand_bbox'] = pose_det_results[idx]['right_hand_bbox'] 
            res['right_hand_valid'] = pose_det_results[idx]['right_hand_valid']
            body_and_hands_lift_results_vis.append(res)
        
        # body_and_hands_lift_results_vis: list[dict()]
        
        if len(body_and_hands_lift_results_vis) == 0:
            continue
        
        if num_instances < 0:
            num_instances = len(body_and_hands_lift_results_vis)

        img_vis = vis_3d_pose_result(
            body_lift_model,
            result=body_and_hands_lift_results_vis,
            img=video[i],
            dataset_info= h3wb_wo_face_dataset_info,
            out_file=None,
            radius=args.radius,
            thickness=args.thickness,
            vis_height=1000,
            num_instances=num_instances,
            show=args.show,
            axis_azimuth=-90)
        
        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    osp.join(args.out_video_root,
                             f'vis_{osp.basename(args.video_path)}'), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)
    
    if save_out_video:
        writer.release()


if __name__ == '__main__':
    main()
    