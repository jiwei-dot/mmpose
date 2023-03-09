# """
#     Copy from https://github.com/davrempe/contact-human-dynamics/blob/main/src/contact_learning/models/openpose_only.py.
# """
# import sys

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import numpy as np

# sys.path.extend(['contact_learning'])
# from data.openpose_dataset import OP_LOWER_JOINTS_MAP


# TORCH_VER = torch.__version__


# class OpenPoseModel(nn.Module):
#     def __init__(self, window_size, joints, pred_size, feat_size):
#         super(OpenPoseModel, self).__init__()
#         self.window_size = window_size
#         self.contact_size = pred_size
#         self.feat_size = feat_size
#         #
#         # Losses
#         #
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
#         self.sigmoid = nn.Sigmoid()

#         #
#         # Create the model
#         #
#         self.model = nn.Sequential(
#                         nn.Linear(window_size * joints * self.feat_size, 1024),
#                         nn.BatchNorm1d(1024),
#                         nn.ReLU(),
                        
#                         nn.Linear(1024, 512),
#                         nn.BatchNorm1d(512),
#                         nn.ReLU(),
                        
#                         nn.Linear(512, 128),
#                         nn.BatchNorm1d(128),
#                         nn.ReLU(),
                        
#                         nn.Dropout(p=0.3),
                        
#                         nn.Linear(128, 32),
#                         nn.BatchNorm1d(32),
#                         nn.ReLU(),
#                         nn.Linear(32, 4 * pred_size)
#                     )
#         # initialize weights
#         self.model.apply(self.init_weights)

#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)

#     def forward(self, x):
#         # data_batch is B x N x J x 3(or 2 if no confidence)
#         B, N, J, F = x.size()
#         # flatten to single data vector
#         x = x.view(B, N*J*F)
#         # run model
#         x = self.model(x)
#         return x.view(B, self.contact_size, 4)

#     def loss(self, outputs, labels):
#         ''' Returns the loss value for the given network output '''
#         B, N, _ = outputs.size()
#         outputs = outputs.view(B, N*4)

#         B, N, _ = labels.size()
#         labels = labels.view(B, N*4)

#         loss_flat = self.bce_loss(outputs, labels)
#         loss = loss_flat.view(B, N, 4)

#         return loss

#     def prediction(self, outputs, thresh=0.5):
#         probs = self.sigmoid(outputs)
#         pred = probs > thresh
#         return pred, probs

#     def accuracy(self, outputs, labels, thresh=0.5, tgt_frame=None):
#         ''' Calculates confusion matrix counts for TARGET (middle) FRAME ONLY'''
#         # threshold to classify
#         pred, _ = self.prediction(outputs, thresh)

#         if tgt_frame is None:
#             tgt_frame = self.contact_size // 2

#         # only want to evaluate accuracy of middle frame
#         pred = pred[:, tgt_frame, :]
#         if TORCH_VER == '1.0.0' or TORCH_VER == '1.1.0':
#             pred = pred.byte()
#         else:
#             # 1.2.0
#             pred = pred.to(torch.bool)
#         labels = labels[:, tgt_frame, :]
#         if TORCH_VER == '1.0.0' or TORCH_VER == '1.1.0':
#             labels = labels.byte()
#         else:
#             labels = labels.to(torch.bool)

#         # counts for confusion matrix
#         # true positive (pred contact, labeled contact)
#         true_pos = pred & labels
#         true_pos_cnt = torch.sum(true_pos).to('cpu').item()
#         # false positive (pred contact, not lebeled contact)
#         false_pos = pred & ~(labels)
#         false_pos_cnt = torch.sum(false_pos).to('cpu').item()
#         # false negative (pred no contact, labeled contact)
#         false_neg = ~(pred) & labels
#         false_neg_cnt = torch.sum(false_neg).to('cpu').item()
#         # true negative (pred no contact, no labeled contact)
#         true_neg = (~pred) & (~labels)
#         true_neg_cnt = torch.sum(true_neg).to('cpu').item()

#         return true_pos_cnt, false_pos_cnt, false_neg_cnt, true_neg_cnt


import warnings

from .. import builder
from ..builder import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class FootContactor(BasePose):
    
    def __init__(self, 
                 backbone, 
                 neck=None,
                 head=None,
                 train_cfg=None, 
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.backbone = builder.build_backbone(backbone)
        
        if neck is not None:
            self.neck = builder.build_neck(neck)
            
        if head is not None:
            head['train_cfg'] = train_cfg
            head['test_cfg'] = test_cfg
            self.head = builder.build_head(head)
            
        self.pretrained = pretrained
        self.init_weights()
        
        
    @property
    def with_neck(self):
        """Check if has keypoint_neck."""
        return hasattr(self, 'neck')


    @property
    def with_head(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'head')
        
        
    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

        
    def forward_train(self, input, target, **kwargs):

        features = self.backbone(input)
        if self.with_neck:
            features = self.neck(features)
        if self.with_head:
            output = self.head(features)

        losses = dict()
        if self.with_head:
            head_losses = self.head.get_loss(output, target)
            head_accuracy = self.head.get_accuracy(output, target)
            losses.update(head_losses)
            losses.update(head_accuracy)
        return losses
        
    
    def forward_test(self, input, metas, **kwargs):
        
        results = {}

        features = self.backbone(input)
        if self.with_neck:
            features = self.neck(features)
        if self.with_head:
            output = self.head.inference_model(features)
            head_result = self.head.decode(output, metas)
            results.update(head_result)
            
        return results
    
    
    @auto_fp16(apply_to=('input', ))
    def forward(self,
                input,
                target=None,
                target_weight=None,
                metas=None,
                return_loss=True,
                **kwargs):
        if return_loss:
            return self.forward_train(input, target, **kwargs)
        else:
            return self.forward_test(input, metas, **kwargs)
        
    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError