# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from mmdet.models.detectors.vfnet import VFNet
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class MultiTaskVFNet(VFNet):
    """Multi-task VFNet with image-level prediction heads.
    
    Adds PH, VBN, and clf_score prediction to standard VFNet detection.
    """
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 image_head: Optional[ConfigType] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        
        # Initialize parent VFNet
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )
        
        # Add image-level head
        if image_head is not None:
            self.image_head = MODELS.build(image_head)
        else:
            self.image_head = None
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract features.
        
        Args:
            batch_inputs (Tensor): Input images
            
        Returns:
            Tuple[Tensor, Tensor]: (neck_features, backbone_final_feature)
        """
        # Get backbone features
        backbone_feats = self.backbone(batch_inputs)
        
        # Get neck features for detection
        if self.with_neck:
            neck_feats = self.neck(backbone_feats)
        else:
            neck_feats = backbone_feats
            
        # Return both neck features (for detection) and final backbone feature (for image-level)
        return neck_feats, backbone_feats[-1]
    
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses.
        
        Args:
            batch_inputs (Tensor): Input images
            batch_data_samples (SampleList): Data samples with annotations
            
        Returns:
            Dict[str, Tensor]: Loss components
        """
        losses = {}
        
        # Extract features
        neck_feats, backbone_final_feat = self.extract_feat(batch_inputs)
        
        # Detection losses (original VFNet)
        det_losses = self.bbox_head.loss(neck_feats, batch_data_samples)
        losses.update(det_losses)
        
        # Image-level losses
        if self.image_head is not None:
            # Get image-level predictions
            img_predictions = self.image_head(backbone_final_feat)
            
            # Extract img_metas from batch_data_samples
            batch_img_metas = []
            for data_sample in batch_data_samples:
                img_meta = data_sample.metainfo
                batch_img_metas.append(img_meta)
            
            # Calculate image-level losses
            img_losses = self.image_head.loss(img_predictions, batch_img_metas)
            losses.update(img_losses)
        
        return losses
    
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, 
                rescale: bool = True) -> SampleList:
        """Predict results.
        
        Args:
            batch_inputs (Tensor): Input images
            batch_data_samples (SampleList): Data samples
            rescale (bool): Whether to rescale results
            
        Returns:
            SampleList: Detection and image-level predictions
        """
        # Extract features
        neck_feats, backbone_final_feat = self.extract_feat(batch_inputs)
        
        # Detection predictions (original VFNet)
        results = self.bbox_head.predict(neck_feats, batch_data_samples, rescale=rescale)
        
        # Image-level predictions
        if self.image_head is not None:
            img_predictions = self.image_head(backbone_final_feat)
            
            # Add image-level predictions to DetDataSample metainfo (not InstanceData)
            for i, result in enumerate(results):
                # Store image-level predictions in metainfo
                result.pred_instances.metainfo = getattr(result.pred_instances, 'metainfo', {})
                result.pred_instances.metainfo['ph_pred'] = img_predictions['ph_pred'][i].item()
                result.pred_instances.metainfo['vbn_pred'] = img_predictions['vbn_pred'][i].item()
                result.pred_instances.metainfo['clf_preds'] = [pred[i].argmax().item() + 1 for pred in img_predictions['clf_preds']]
        
        return results 