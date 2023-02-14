import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import (build_upsample_layer, build_conv_layer, build_norm_layer, normal_init, constant_init)
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from mmpose.core.evaluation import pose_pck_accuracy, keypoint_mpjpe
from mmpose.core.post_processing import flip_back
from mmpose.core.camera import SimpleCamera

from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from ..builder import HEADS


@HEADS.register_module()
class TopdownHeatmapSimpleDoubleHeads(TopdownHeatmapBaseHead):
    def __init__(self,
                 in_channels,
                 out_channels,
                 
                 # one branch
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 
                 # another branch
                 num_conv_layers=3,
                 num_conv_filters=(256, 256, 256),
                 num_conv_kernels=(3, 3, 3),
                 
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint_list=None,
                 train_cfg=None,
                 test_cfg=None):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert len(loss_keypoint_list) == 2
        self.loss1 = build_loss(loss_keypoint_list[0])
        self.loss2 = build_loss(loss_keypoint_list[1])
        
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
        
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners
        
        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')
        
        # one branch
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        
        
        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0
            
        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels
            
            layers = []
            
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels', [1] * num_conv_layers)
                
                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))
                    
            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            
            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

        self.one_branch = nn.Sequential(
            self.deconv_layers,
            self.final_layer
        )

        # another branch
        if num_conv_layers > 0:
            self.conv_layers = self._make_conv_layer(
                num_conv_layers,
                num_conv_filters,
                num_conv_kernels,
            )
        elif num_conv_layers == 0:
            self.conv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_conv_layers ({num_conv_layers}) should >= 0.')
            
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(2, 2))
        

        self.final_conv = nn.Conv2d(
            num_conv_filters[-1] if num_conv_layers > 0 else (self.in_channels + out_channels),
            out_channels, 1, 1)
        
        self.final_fc = nn.Linear(4, 3)
            
        self.another_branch = nn.Sequential(
            self.conv_layers,
            self.maxpool,
            self.final_conv,
        )

    def get_loss(self, output, target, target_weight, xyz_camera_space, xyz_weight):
        assert len(output) == 2
        losses = dict()
        losses['heatmap_loss'] = self.loss1(output[0], target, target_weight) * 100
        losses['xyz_loss'] = self.loss2(output[1], xyz_camera_space, xyz_weight)
        return losses
    
    def get_accuracy(self, output, target, target_weight, xyz_camera_space, xyz_weight):
        assert len(output) == 2
        # output[0]: B x N x H x W
        # output[1]: B x N x 3
        accuracy = dict()
        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output[0].detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['xy_acc_pose'] = float(avg_acc)
        
        mask = (xyz_weight > 0).detach().cpu().numpy().squeeze(axis=-1)
        accuracy['xyz_mpjpe'] = float(keypoint_mpjpe(
            output[1].detach().cpu().numpy(),
            xyz_camera_space.detach().cpu().numpy(),
            mask))
        # print(accuracy)
        return accuracy
    
    def forward(self, x):
        x = self._transform_inputs(x)           # [B, 256, 64, 48]
        B = x.shape[0]
        out1 = self.one_branch(x)               # [B, 17, 64, 48]
        x = torch.cat([x, out1], dim=1)         # [B, 256+17, 64, 48]
        out2 = self.another_branch(x)           # [B, 17, 2, 2]
        N = out2.shape[1]
        out2 = out2.reshape(B, N, -1)           # [B, 17, 4]
        out2 = self.final_fc(out2)              # [B, 17, 3]        # 预测绝对坐标，后期改成相对
        return [out1, out2]
    
    def inference_model(self, x, flip_pairs):
        assert flip_pairs is None, 'Not implemented when flip_pairs it not None'
        output = self.forward(x)
        assert len(output) == 2
        # if flip_pairs is not None:
        #     output_heatmap = flip_back(
        #         output[0].detach().cpu().numpy(),
        #         flip_pairs,
        #         target_type=self.target_type)
        #     # feature is not aligned, shift flipped heatmap for higher accuracy
        #     if self.test_cfg.get('shift_heatmap', False):
        #         output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        # else:
        #     output_heatmap = output.detach().cpu().numpy()
        return [output[0].detach().cpu().numpy(), output[1].detach().cpu().numpy()]    
    
    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels
    
    def _transform_inputs(self, inputs):
        
        if not isinstance(inputs, list):
            return inputs
        
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        inplanes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=inplanes,
                    out_channels=num_filters[i],
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            inplanes = num_filters[i]

        return nn.Sequential(*layers)
    
    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        inplanes = self.in_channels + self.out_channels
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=num_filters[i],
                    kernel_size=num_kernels[i],
                    stride=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            inplanes = num_filters[i]
            
        return nn.Sequential(*layers)
    
    def init_weights(self):
        """Initialize model weights."""
        
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    
        for _, m in self.conv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
                
    def decode(self, img_metas, output, **kwargs):
        assert len(output) == 2
        # output[0]: B x N x H x W
        # output[1]: B x N x 3
        batch_size = len(img_metas)
        results = super().decode(img_metas, output[0], **kwargs)
        # preds, boxes, image_paths, bbox_ids
        
        results['preds_xy_pixel'] = results['preds']
        del results['preds']
        # preds_xy_pixel, boxes, image_paths, bbox_ids
        
        # B x N x 3
        preds_xyz_camera = np.zeros((batch_size, output[0].shape[1], 3), dtype=np.float32)
        
        for i in range(batch_size):
            preds_xyz_camera[i][:, :3] = output[1][i]
            
        results['preds_xyz_camera'] = preds_xyz_camera
        
        return results
            
        