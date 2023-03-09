# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import build_conv_layer, constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.models.builder import HEADS


@HEADS.register_module()
class CustomTemporalClassificationHead(nn.Module):

    def __init__(self,
                 in_channels,
                 reduction=8,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        mid_channels = max(8, in_channels // reduction)
        
        self.conv = build_conv_layer(
            dict(type='Conv1d'), in_channels, mid_channels, 1)
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.final_layer = build_conv_layer(
            dict(type='Conv1d'), mid_channels, 4, 1)

    @staticmethod
    def _transform_inputs(x):
        if not isinstance(x, (list, tuple)):
            return x
        assert len(x) > 0
        return x[-1]

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)

        assert x.ndim == 3 and x.shape[2] == 1, f'Invalid shape {x.shape}'
        output = self.relu(self.bn(self.conv(x)))
        output = self.final_layer(output)
        N = output.shape[0]
        return output.reshape(N, 4)

    def get_loss(self, output, target):
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        target = target.to(dtype=output.dtype, device=output.device)
        losses['cls_loss'] = self.loss(output, target)
        return losses

    def get_accuracy(self, output, target):
        # output: [N, 4]
        # target: [N, 4]
        accuracy = dict()
        
        output = output.detach().cpu().sigmoid()
        target = target.detach().cpu()
        pred = (output > 0.5).to(dtype=target.dtype)
        
        acc = (pred == target).sum() / pred.numel()
        accuracy['acc'] = acc.item()
        return accuracy

    def inference_model(self, x):
        output = self.forward(x)
        output_classification = output.detach().cpu().sigmoid().numpy()
        return output_classification

    def decode(self, output, metas):
        output = np.array(output > 0.5, dtype=np.float32)
        target_image_paths = [m.get('imgname', None) for m in metas]
        result = {'preds': output, 'target_image_paths': target_image_paths}
        return result

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
