# Author: Ji, Wei

import os.path as osp
import tempfile
from collections import OrderedDict

import numpy as np

import mmcv
from mmcv import deprecated_api_warning
from mmpose.core.evaluation import (keypoint_3d_auc, keypoint_3d_pck, keypoint_mpjpe)
from mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from ...builder import DATASETS


@DATASETS.register_module()
class Hand3DH3WBDataset(Kpt3dSviewKpt2dDataset):
    
    JOINT_NAMES = []
    
    SUPPORTED_JOINT_2D_SRC = {
        'gt', 'detection', 'pipeline'
    }
    
    ALLOWED_METRICS = {
        'mpjpe', 'p-mpjpe', '3dpck', 'p-3dpck', '3dauc', 'p-3dauc'
    }

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        
        if dataset_info is None:
            raise ValueError

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # mpi-inf-3dhp specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])

        # mpi-inf-3dhp specific annotation info
        ann_info = {}
        ann_info['use_different_joint_weights'] = False

        self.ann_info.update(ann_info)

    def load_annotations(self):
        data_info = super().load_annotations()

        # get 2D joints
        if self.joint_2d_src == 'gt':
            data_info['joints_2d'] = data_info['joints_2d']
        elif self.joint_2d_src == 'detection':
            data_info['joints_2d'] = self._load_joint_2d_detection(
                self.joint_2d_det_file)
            assert data_info['joints_2d'].shape[0] == data_info[
                'joints_3d'].shape[0]
            assert data_info['joints_2d'].shape[2] == 3
        elif self.joint_2d_src == 'pipeline':
            # joint_2d will be generated in the pipeline
            pass
        else:
            raise NotImplementedError(
                f'Unhandled joint_2d_src option {self.joint_2d_src}')

        return data_info

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """
        return [[i] for i in range(len(self.data_info['imgnames']))]

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)

        return joints_2d

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for mpi-inf-3dhp dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            image_paths = result['target_image_paths']
            batch_size = len(image_paths)
            for i in range(batch_size):
                target_id = self.name2id[image_paths[i]]
                kpts.append({
                    'keypoints': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(kpts)
            elif _metric == 'p-3dpck':
                _nv_tuples = self._report_3d_pck(kpts, mode='p-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(kpts)
            elif _metric == 'p-3dauc':
                _nv_tuples = self._report_3d_auc(kpts, mode='p-3dauc')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants
        P-MPJPE.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:
                - ``'mpjpe'``: Standard MPJPE.
                - ``'p-mpjpe'``: MPJPE after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
        """

        preds = []
        gts = []
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)

        preds = np.stack(preds)
        gts = np.stack(gts)
        masks = np.ones_like(gts[:, :, 0], dtype=bool)

        err_name = mode.upper()
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error = keypoint_mpjpe(preds, gts, masks, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_3d_pck(self, keypoint_results, mode='3dpck'):
        """Cauculate Percentage of Correct Keypoints (3DPCK) w. or w/o
        Procrustes alignment.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:
                - ``'3dpck'``: Standard 3DPCK.
                - ``'p-3dpck'``: 3DPCK after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
        """

        preds = []
        gts = []
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)

        preds = np.stack(preds)
        gts = np.stack(gts)
        masks = np.ones_like(gts[:, :, 0], dtype=bool)

        err_name = mode.upper()
        if mode == '3dpck':
            alignment = 'none'
        elif mode == 'p-3dpck':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error = keypoint_3d_pck(preds, gts, masks, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_3d_auc(self, keypoint_results, mode='3dauc'):
        """Cauculate the Area Under the Curve (AUC) computed for a range of
        3DPCK thresholds.

        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:

                - ``'3dauc'``: Standard 3DAUC.
                - ``'p-3dauc'``: 3DAUC after aligning prediction to
                    groundtruth via a rigid transformation (scale, rotation and
                    translation).
        """

        preds = []
        gts = []
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)

        preds = np.stack(preds)
        gts = np.stack(gts)
        masks = np.ones_like(gts[:, :, 0], dtype=bool)

        err_name = mode.upper()
        if mode == '3dauc':
            alignment = 'none'
        elif mode == 'p-3dauc':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error = keypoint_3d_auc(preds, gts, masks, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _load_camera_param(self, camear_param_file):
        """Load camera parameters from file."""
        return mmcv.load(camear_param_file)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        return self.camera_param[imgname[:-11]]


@DATASETS.register_module()
class DoubleHands3DH3WBDataset(Hand3DH3WBDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        
        self.larm_indices = [5, 7, 9]
        self.rarm_indices = [6, 8, 10]
        self.lhand_indices = list(range(91, 112))
        self.rhand_indices = list(range(112, 133)) 
        self.hand_indices = self.lhand_indices + self.rhand_indices
        self.all_indices = self.larm_indices + self.rarm_indices + self.lhand_indices + self.rhand_indices
        
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        
    def prepare_data(self, idx):
        data = self.data_info
        frame_ids = self.sample_indices[idx]
        assert len(frame_ids) == self.seq_len

        # get the 3D/2D pose sequence
        # [seq_len, 48, 4]
        _joints_3d = data['joints_3d'][frame_ids]
        
        # [seq_len, 48, 3]
        _joints_2d = data['joints_2d'][frame_ids]

        # get the image info
        _imgnames = data['imgnames'][frame_ids]
        _centers = data['centers'][frame_ids]
        _scales = data['scales'][frame_ids]
        if _scales.ndim == 1:
            _scales = np.stack([_scales, _scales], axis=1)

        target_idx = -1 if self.causal else int(self.seq_len) // 2

        results = {
            # [seq_len, 48, 2]
            'input_2d': _joints_2d[:, :, :2],
            # [seq_len, 48, 1]
            'input_2d_visible': _joints_2d[:, :, -1:],
            # [seq_len, 48, 3]
            'input_3d': _joints_3d[:, :, :3],
            # [seq_len, 48, 1]
            'input_3d_visible': _joints_3d[:, :, -1:],
            
            # 'target_idx': target_idx,
            
            # [48, 3]
            'target': _joints_3d[target_idx, :, :3],
            # [48, 1]
            'target_visible': _joints_3d[target_idx, :, -1:],
            
            'image_paths': _imgnames,
            'target_image_path': _imgnames[target_idx],
            'scales': _scales,
            'centers': _centers,
        }
        return results
    
    def load_annotations(self):
        """Load data annotation."""
        data = np.load(self.ann_file)

        # get image info
        _imgnames = data['imgname']
        num_imgs = len(_imgnames)
        num_joints = self.ann_info['num_joints']

        if 'scale' in data:
            _scales = data['scale'].astype(np.float32)
        else:
            _scales = np.zeros(num_imgs, dtype=np.float32)

        if 'center' in data:
            _centers = data['center'].astype(np.float32)
        else:
            _centers = np.zeros((num_imgs, 2), dtype=np.float32)

        # get 3D pose
        if 'S' in data.keys():
            _joints_3d = data['S'].astype(np.float32)
        else:
            _joints_3d = np.zeros((num_imgs, num_joints, 4), dtype=np.float32)

        # get 2D pose
        if 'part' in data.keys():
            _joints_2d = data['part'].astype(np.float32)
        else:
            _joints_2d = np.zeros((num_imgs, num_joints, 3), dtype=np.float32)

        data_info = {
            'imgnames': _imgnames,
            'joints_3d': _joints_3d[:, self.all_indices],
            'joints_2d': _joints_2d[:, self.all_indices],
            'scales': _scales,
            'centers': _centers,
        }

        return data_info
    

@DATASETS.register_module()
class LocalAndGlobalHand3DH3WBDataset(Hand3DH3WBDataset):
    
    def __init__(self, 
                 ann_file, 
                 img_prefix, 
                 data_cfg, 
                 pipeline, 
                 dataset_info=None, 
                 test_mode=False):
        
        super().__init__(ann_file, img_prefix, data_cfg, pipeline, dataset_info, test_mode)
        
        
    def prepare_data(self, idx):
        data = self.data_info
        frame_ids = self.sample_indices[idx]
        assert len(frame_ids) == self.seq_len

        # get the 3D/2D pose sequence
        # [seq_len, 22, 4]
        _joints_3d = data['joints_3d'][frame_ids]
        
        # [seq_len, 22, 3]
        _joints_2d = data['joints_2d'][frame_ids]

        # get the image info
        _imgnames = data['imgnames'][frame_ids]
        _centers = data['centers'][frame_ids]
        _scales = data['scales'][frame_ids]
        if _scales.ndim == 1:
            _scales = np.stack([_scales, _scales], axis=1)

        target_idx = -1 if self.causal else int(self.seq_len) // 2

        results = {
            # [seq_len, 22, 2]
            'input_2d': _joints_2d[:, :, :2],
            # [seq_len, 22, 1]
            'input_2d_visible': _joints_2d[:, :, -1:],
            # [seq_len, 22, 3]
            'input_3d': _joints_3d[:, :, :3],
            # [seq_len, 22, 1]
            'input_3d_visible': _joints_3d[:, :, -1:],
            
            # [22, 3]
            'target': _joints_3d[target_idx, :, :3],
            # [22, 1]
            'target_visible': _joints_3d[target_idx, :, -1:],
            
            'image_paths': _imgnames,
            'target_image_path': _imgnames[target_idx],
            'scales': _scales,
            'centers': _centers,
        }
        return results