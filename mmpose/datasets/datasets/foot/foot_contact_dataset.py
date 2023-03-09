import os.path as osp
import tempfile
from collections import defaultdict, OrderedDict
import copy
from torch.utils.data import Dataset
import numpy as np
from ...builder import DATASETS
from mmpose.datasets.pipelines import Compose
import mmcv
from mmcv import deprecated_api_warning


@DATASETS.register_module()
class FootContactDataset(Dataset):
    
    ALLOWED_METRICS = {'precision', 'recall', 'f1_score'}
    
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.data_cfg = data_cfg
        self.pipeline = pipeline
        self.test_mode = test_mode
        
        self.load_config(self.data_cfg)
        self.data_info = self.load_annotations()
        self.sample_indices = self.build_sample_indices()
        self.pipeline = Compose(pipeline)
        
        self.name2id = {
            name: i
            for i, name in enumerate(self.data_info['imgnames'])
        }
  
        
    def load_config(self, data_cfg):
        self.seq_len = data_cfg.get('seq_len', 9)
        self.seq_frame_interval = data_cfg.get('seq_frame_interval', 1)
        self.causal = data_cfg.get('causal', False)
        self.temporal_padding = data_cfg.get('temporal_padding', True)
    
    
    def __len__(self):
        return len(self.sample_indices)
    
    
    def __getitem__(self, idx):
        results = copy.deepcopy(self.prepare_data(idx))
        return self.pipeline(results)
    
    
    def prepare_data(self, idx):
        data = self.data_info
        frame_ids = self.sample_indices[idx]
        assert len(frame_ids) == self.seq_len
        
        # [9, 13, 3]
        kpts2d = data['kpts2d'][frame_ids]
        # [9, 4]
        contacts = data['contacts'][frame_ids]
        imgnames = data['imgnames'][frame_ids]
        
        target_idx = -1 if self.causal else int(self.seq_len) // 2
        
        results = {
            'imgname': imgnames[target_idx],
            'input_2d': kpts2d,
            'target': contacts[target_idx]
        }
        return results
    
    
    def load_annotations(self):
        data = np.load(self.ann_file)
        data_info = {
            'imgnames': data['imgnames'],
            'kpts2d': data['kpts2d'],
            'contacts': data['contacts']
        }
        return data_info
    
    
    def build_sample_indices(self):
        video_frames = defaultdict(list)
        for idx, imgname in enumerate(self.data_info['imgnames']):
            character, action, view = self._parse_imgname(imgname)
            video_frames[(character, action, view)].append(idx)
            
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)
            
            if self.temporal_padding:
                # Pad the sequence so that every frame in the sequence will be
                # predicted.
                if self.causal:
                    frames_left = self.seq_len - 1
                    frames_right = 0
                else:
                    frames_left = (self.seq_len - 1) // 2
                    frames_right = frames_left
                for i in range(n_frame):
                    pad_left = max(0, frames_left - i // _step)
                    pad_right = max(0, frames_right - (n_frame - 1 - i) // _step)
                    start = max(i % _step, i - frames_left * _step)
                    end = min(n_frame - (n_frame - 1 - i) % _step,
                              i + frames_right * _step + 1)
                    sample_indices.append([_indices[0]] * pad_left +
                                          _indices[start:end:_step] +
                                          [_indices[-1]] * pad_right)
            else:
                seqs_from_video = [
                    _indices[i:(i + _len):_step]
                    for i in range(0, n_frame - _len + 1)
                ]
                sample_indices.extend(seqs_from_video)
                
        return sample_indices
            
    
    @staticmethod
    def _parse_imgname(imgname):
        return imgname.split('/')[:3]
    
    
    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='f1_score', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
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
                    'preds': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)
        
        name_value_tuples = []
        
        confusion_matrix = np.zeros((2, 2), dtype=np.int64)
        
        for kpt in kpts:
            pred = kpt['preds']
            gt = self.data_info['contacts'][kpt['target_id']]

            num_true_neg = ((pred == 0) & (gt == 0)).sum()
            num_false_neg = ((pred == 0) & (gt == 1)).sum()
            num_true_pos = ((pred == 1) & (gt == 1)).sum()
            num_false_pos = ((pred == 1) & (gt == 0)).sum()
            
            assert (num_true_neg + num_false_neg + num_true_pos + num_false_pos) == 4
            confusion_matrix[0][0] += num_true_neg
            confusion_matrix[0][1] += num_false_neg
            confusion_matrix[1][0] += num_false_pos
            confusion_matrix[1][1] += num_true_pos
            
        precision = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        recall = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
        f1_score = 2 * precision * recall / (precision + recall + 1e-5)
        
        name_value_tuples.append(('precision', precision))
        name_value_tuples.append(('recall', recall))
        name_value_tuples.append(('f1_score', f1_score))
            
        
        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)