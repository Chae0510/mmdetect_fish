# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from mmdet.models.detectors.vfnet import VFNet
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class BboxLevelVFNet(VFNet):
    """Bbox-level VFNet for detecting objects with bbox-level attributes.
    
    Unlike image-level predictions, this detector predicts ph, vbn, and clf_score
    for each detected bounding box individually.
    """
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        
        # Initialize parent VFNet - no modifications needed
        # The bbox_head should be BboxLevelVFNetHead which handles the attributes
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
    
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses including bbox-level attribute losses.
        
        Args:
            batch_inputs (Tensor): Input images
            batch_data_samples (SampleList): Data samples with bbox-level annotations
            
        Returns:
            Dict[str, Tensor]: Loss components including attribute losses
        """
        # Extract features using parent method
        x = self.extract_feat(batch_inputs)
        
        # Calculate all losses in the bbox_head (detection + attributes)
        losses = self.bbox_head.loss(x, batch_data_samples)
        
        return losses
    
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, 
                rescale: bool = True) -> SampleList:
        """Predict results including bbox-level attributes.
        
        Args:
            batch_inputs (Tensor): Input images
            batch_data_samples (SampleList): Data samples
            rescale (bool): Whether to rescale results
            
        Returns:
            SampleList: Detection results with bbox-level attribute predictions
        """
        # Extract features using parent method
        x = self.extract_feat(batch_inputs)
        
        # Get predictions from bbox_head (includes attribute predictions)
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        
        # Follow parent class pattern: add predictions to data samples
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        
        return batch_data_samples 