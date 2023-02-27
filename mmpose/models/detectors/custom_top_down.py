# Copyright (c) OpenMMLab. All rights reserved.
import warnings


from ..builder import POSENETS
from .top_down import TopDown

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class CustomTopDown(TopDown):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__(
            backbone,
            neck,
            keypoint_head,
            train_cfg,
            test_cfg,
            pretrained,
            loss_pose)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target_xy=None,
                target_xy_weight=None,
                target_z=None,
                target_z_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        if return_loss:
            return self.forward_train(img, target_xy, target_xy_weight, 
                                      target_z, target_z_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target_xy, target_xy_weight, 
                      target_z, target_z_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output,target_xy, target_xy_weight, target_z, target_z_weight, **kwargs)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target_xy, target_xy_weight, target_z, target_z_weight, **kwargs)
            losses.update(keypoint_accuracy)
       
        return losses

    # def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
    #     """Defines the computation performed at every call when testing."""
    #     assert img.size(0) == len(img_metas)
    #     batch_size, _, img_height, img_width = img.shape
    #     # if batch_size > 1:
    #     #     assert 'bbox_id' in img_metas[0]

    #     result = {}

    #     features = self.backbone(img)
    #     if self.with_neck:
    #         features = self.neck(features)
    #     if self.with_keypoint:
    #         output_heatmap = self.keypoint_head.inference_model(
    #             features, flip_pairs=None)

    #     if self.test_cfg.get('flip_test', True):
    #         img_flipped = img.flip(3)
    #         features_flipped = self.backbone(img_flipped)
    #         if self.with_neck:
    #             features_flipped = self.neck(features_flipped)
    #         if self.with_keypoint:
    #             output_flipped_heatmap = self.keypoint_head.inference_model(
    #                 features_flipped, img_metas[0]['flip_pairs'])
    #             output_heatmap = (output_heatmap + output_flipped_heatmap)
    #             if self.test_cfg.get('regression_flip_shift', False):
    #                 output_heatmap[..., 0] -= 1.0 / img_width
    #             output_heatmap = output_heatmap / 2

    #     if self.with_keypoint:
    #         keypoint_result = self.keypoint_head.decode(
    #             img_metas, output_heatmap, img_size=[img_width, img_height])
    #         result.update(keypoint_result)

    #         if not return_heatmap:
    #             output_heatmap = None

    #         result['output_heatmap'] = output_heatmap

    #     return result


   