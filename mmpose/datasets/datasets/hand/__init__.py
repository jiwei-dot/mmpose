# Copyright (c) OpenMMLab. All rights reserved.
from .freihand_dataset import FreiHandDataset
from .hand_coco_wholebody_dataset import HandCocoWholeBodyDataset
from .interhand2d_dataset import InterHand2DDataset
from .interhand3d_dataset import InterHand3DDataset
from .onehand10k_dataset import OneHand10KDataset
from .panoptic_hand2d_dataset import PanopticDataset
from .rhd2d_dataset import Rhd2DDataset
from .h3wbhand_dataset import Hand3DH3WBDataset
from .hand_2dand3d_dataset import Hand2DAnd3DMergeDataset


__all__ = [
    'FreiHandDataset', 'InterHand2DDataset', 'InterHand3DDataset',
    'OneHand10KDataset', 'PanopticDataset', 'Rhd2DDataset',
    'HandCocoWholeBodyDataset', 'Hand3DH3WBDataset',
    'Hand2DAnd3DMergeDataset'
]
