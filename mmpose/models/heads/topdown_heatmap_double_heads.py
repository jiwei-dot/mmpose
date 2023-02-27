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
class TopdownHeatmapDoubleHeads(TopdownHeatmapBaseHead):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 
                 # two heads share same
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1,
                 
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
    
        if num_deconv_layers > 0:
            self.head1_deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels)
            self.head2_deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels)
        elif num_deconv_layers == 0:
            self.head1_deconv_layers = nn.Identity()
            self.head2_deconv_layers = nn.Identity()
        else:
            raise ValueError(f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        
        conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels      
        
        self.final_layer1 = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=conv_channels,
            out_channels=out_channels,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=0)
        
        self.final_layer2 = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=conv_channels,
            out_channels=out_channels,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=0)
        
        self.final_fc = nn.Linear(out_channels, out_channels-1)

    def get_loss(self, output, target_xy, target_xy_weight, target_z, target_z_weight):
        assert len(output) == 2
        losses = dict()
        # print('*' * 100)
        # print(target_z)
        # print(target_z_weight)
        # print('*' * 100)
        losses['heatmap_loss'] = self.loss1(output[0], target_xy, target_xy_weight)
        losses['rel_z_loss'] = self.loss2(output[1], target_z, target_z_weight)
        # print(losses)
        return losses
    
    def get_accuracy(self, output, target_xy, target_xy_weight, target_z, target_z_weight):
        assert len(output) == 2
        # output[0]: B x N x H x W
        # output[1]: B x N
        
        # print(target_xy_weight.shape)   # B x N x 1
        # print(target_z_weight.shape)    # B x N
        
        target_xy_weight = target_xy_weight.detach().cpu().numpy().squeeze(axis=-1)
        xy_mask = target_xy_weight > 0
        
        accuracy = dict()
        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output[0].detach().cpu().numpy(),
                target_xy.detach().cpu().numpy(),
                xy_mask)
            accuracy['xy_acc_pose'] = float(avg_acc)
        
        # todo
        # mask = (xyz_weight > 0).detach().cpu().numpy().squeeze(axis=-1)
        # accuracy['xyz_mpjpe'] = float(keypoint_mpjpe(
        #     output[1].detach().cpu().numpy(),
        #     xyz_camera_space.detach().cpu().numpy(),
        #     mask))
        # # print(accuracy)
        return accuracy
    
    def forward(self, x):
        feat = self._transform_inputs(x)              # [B, 256, 64, 64]
        head1_out = self.final_layer1(self.head1_deconv_layers(feat))               # [B, 21, 64, 64]
        feat = self.final_layer2(self.head2_deconv_layers(feat)) # [B, 21, 64, 64]
        
        # B, N, H, W = head1_out.shape
        # tmp = head1_out.reshape(B*N, -1)
        # weight = torch.exp(tmp) / torch.exp(tmp).sum(dim=1, keepdim=True)
        # weight = weight.reshape(B, N, H, W)
     
        feat = (head1_out * feat).sum(dim=[-2, -1])   # [B, 22]
        head2_out = self.final_fc(feat)

        return [head1_out, head2_out]
    
    def inference_model(self, x, flip_pairs):
        assert flip_pairs is None, 'Not implemented when flip_pairs it not None'
        outputs_list = self.forward(x)
        assert len(outputs_list) == 2
        head1_out, head2_out = outputs_list
        return [head1_out.detach().cpu().numpy(), head2_out.detach().cpu().numpy()]    
     
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
        
        for _, m in self.head1_deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        for _, m in self.final_layer1.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        for _, m in self.head2_deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        for _, m in self.final_layer2.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        normal_init(self.final_fc, mean=0, std=0.001)
           
    def decode(self, img_metas, output, **kwargs):
        assert len(output) == 2
        # output[0]: B x N x H x W
        # output[1]: B x (N-1)

        results = super().decode(img_metas, output[0], **kwargs)
        # preds, boxes, image_paths, bbox_ids
        
        batch_size, num_keypoints = output[0].shape[:2]
        
        # B x N x 3
        preds_xy_pixel = results['preds']
        del results['preds']
        
        # B x N-1
        preds_z_rel = output[1]
        preds_z_cam = np.zeros((batch_size, num_keypoints), dtype=np.float32)
        for idx in range(batch_size):
            preds_z_cam[idx][0] = img_metas[idx]['abs_depth']
            preds_z_cam[idx][1:] = preds_z_rel[idx] + img_metas[idx]['abs_depth']
            
            
        # B x N x 4   (x_p, y_p, z_c, score)
        preds = np.zeros((batch_size, num_keypoints, 4), dtype=np.float32)
        preds[..., :2] = preds_xy_pixel[..., :2]
        preds[..., 2] = preds_z_cam
        preds[..., 3:4] = preds_xy_pixel[..., 2:3]
        
        results['preds'] = preds
        
        return results
            
        