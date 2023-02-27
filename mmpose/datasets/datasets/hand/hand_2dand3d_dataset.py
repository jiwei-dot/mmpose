import numpy as np
import os.path as osp
import tempfile
import json
from collections import OrderedDict

from mmcv import deprecated_api_warning
from mmpose.datasets.builder import DATASETS
from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_pck_accuracy)
from mmpose.core.evaluation.pose3d_eval import keypoint_mpjpe
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class Hand2DAnd3DMergeDataset(Kpt2dSviewRgbImgTopDownDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info,
                 test_mode):
        
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info,
            coco_style=True,
            test_mode=test_mode)
        
        self.db = self._get_db()
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')
    
    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)
            for obj in objs:
                image_file = osp.join(self.img_prefix, self.id2name[img_id])
                rotation = 0
                dataset = self.dataset_name
                bbox_id = bbox_id
                bbox_score = 1
                bbox = obj['bbox']
                
                keypoints_2d = np.array(obj['keypoints_2d']).reshape(-1, 3)
                if obj['keypoints_3d'] is not None:
                    keypoints_3d = np.array(obj['keypoints_3d']).reshape(-1, 4)
                else:
                    keypoints_3d = np.zeros((num_joints, 4), dtype=np.float32)
                
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d[:, :2] = keypoints_2d[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints_2d[:, 2:3])
                
                joints_4d = np.zeros((num_joints, 4), dtype=np.float32)
                joints_4d_visible = np.zeros((num_joints, 4), dtype=np.float32)
                if obj['keypoints_3d'] is not None:
                    joints_4d[:, :3] = keypoints_3d[:, :3]
                    joints_4d_visible[:, :3] = np.minimum(1, keypoints_3d[:, 3:4])
                
                gt_db.append({
                    'image_file': image_file,
                    'rotation': rotation,
                    'dataset': dataset,
                    'bbox_id': bbox_id,
                    'bbox': bbox,
                    'bbox_score': bbox_score,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'joints_4d': joints_4d,
                    'joints_4d_visible': joints_4d_visible,
                    'abs_depth_valid': (obj['keypoints_3d'] is not None),
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])
        return gt_db          
        
    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric=None, **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'MPJPE', 'PMPJPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        
        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')
            
        kpts = []
        for result in results:
            # [B, N, 4]
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']
            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                
                kpts.append({
                    'keypoints_xy': preds[i][:, [0, 1, 3]].tolist(),
                    'keypoints_z': preds[i][:, [2]].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value
    
    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        
        outputs_3d = []
        gts_3d = []
        
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []
        

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints_xy'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            
            if item.get('abs_depth_valid', False):
                outputs_3d.append(
                    np.column_stack([
                        item['joints_4d'][:, :2],
                        np.array(pred['keypoints_z'])
                    ]))
                gts_3d.append(item['joints_4d'][:, :3])
            else:
                # 全0处理
                outputs_3d.append(item['joints_4d'][:, :3])
                gts_3d.append(item['joints_4d'][:, :3])
                
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        
        outputs_3d = np.array(outputs_3d)
        gts_3d = np.array(gts_3d)
        
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('PCK', pck))


        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
                                                 auc_nor)))

        if 'MPJPE' in metrics and len(outputs_3d) != 0:
            info_str.append(('MPJPE', keypoint_mpjpe(outputs_3d, gts_3d, masks, 'none')))
            
        if 'PMPJPE' in metrics and len(outputs_3d) != 0:
            info_str.append(('PMPJPE', keypoint_mpjpe(outputs_3d, gts_3d, masks, 'procrustes')))


        return info_str
    
    
    def get_camera_intrisic(self, joints_3d, joints_4d):
        X = joints_3d[:, :2]
        y = joints_4d[:, :2] / joints_4d[:, 2:3]
        
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