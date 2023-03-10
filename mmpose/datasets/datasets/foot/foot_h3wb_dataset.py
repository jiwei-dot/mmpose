import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import deprecated_api_warning

from mmpose.core.evaluation import (keypoint_3d_auc, keypoint_3d_pck,
                                    keypoint_mpjpe)
from mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from ...builder import DATASETS


@DATASETS.register_module()
class Foot3DH3WBDataset(Kpt3dSviewKpt2dDataset):

    JOINT_NAMES = ['left_ankle', 'left_big_toe', 'left_small_toe',  'left_heel',
                   'right_ankle', 'right_big_toe', 'right_small_toe',  'right_heel']
    
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

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
            raise ValueError('dataset_info must be provided')

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