# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList


@MODELS.register_module()
class SimpleClfHead(BaseModule):
    """Simple classification head without bbox prediction.
    
    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
        loss_weight (float): Loss weight.
    """
    
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 loss_weight: float = 1.0,
                 init_cfg: ConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        
        # Simple classification conv
        self.cls_conv = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Forward function.
        
        Args:
            x (List[Tensor]): Features from FPN.
            
        Returns:
            List[Tensor]: Classification scores.
        """
        cls_scores = []
        for feat in x:
            cls_score = self.cls_conv(feat)
            cls_scores.append(cls_score)
        return cls_scores
    
    def loss_by_feat(self, cls_scores, batch_gt_instances, batch_img_metas):
        """Calculate loss for classification head.
        
        Args:
            cls_scores (list[Tensor]): Classification scores for each FPN level
            batch_gt_instances: Ground truth instances (not used for image-level tasks)
            batch_img_metas (list[dict]): Meta information for each image
            
        Returns:
            dict: Loss dictionary
        """
        # Process each image in the batch
        total_loss = 0.0
        valid_samples = 0
        
        for img_idx, img_meta in enumerate(batch_img_metas):
            clf_score = img_meta.get('clf_score', None)
            
            if clf_score is None:
                continue
                
            # clf_score is a list of 8 values, we need the value for this specific head
            # The head index should be passed in somehow, for now let's use head_idx from self
            if hasattr(self, 'head_idx'):
                head_idx = self.head_idx
            else:
                # If head_idx not set, we'll need to get it from the calling context
                # For now, let's assume it's the first head (index 0) and add proper indexing later
                head_idx = 0
                
            if head_idx >= len(clf_score):
                continue
                
            target_value = clf_score[head_idx]
            
            # Get corresponding prediction for this image
            # cls_scores[0] has shape [batch_size, num_classes, H, W]
            # We need to pool it to get image-level prediction
            pred_for_img = cls_scores[0][img_idx]  # [num_classes, H, W]
            
            # Global average pooling to get image-level features
            pred_value = pred_for_img.mean(dim=[1, 2])  # [num_classes]
            
            # Convert target to tensor and adjust for 0-indexed classes
            # clf_score values are 1-5, but we need 0-4 for classes
            target_tensor = torch.tensor(target_value - 1, dtype=torch.long, device=pred_value.device)
            
            # Ensure target is within valid range
            target_tensor = torch.clamp(target_tensor, 0, self.num_classes - 1)
            
            # Calculate cross-entropy loss for this sample
            loss = F.cross_entropy(pred_value.unsqueeze(0), target_tensor.unsqueeze(0))
            total_loss += loss
            valid_samples += 1
            
        if valid_samples == 0:
            # No valid samples, return zero loss
            device = cls_scores[0].device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            total_loss = total_loss / valid_samples
            
        return {'clf_loss': total_loss * self.loss_weight}
    
    def predict_by_feat(self,
                       cls_scores: List[Tensor],
                       batch_img_metas: List[dict],
                       rescale: bool = True) -> InstanceList:
        """Predict results.
        
        Args:
            cls_scores (List[Tensor]): Classification scores.
            batch_img_metas (List[dict]): Meta information.
            rescale (bool): Whether to rescale results.
            
        Returns:
            InstanceList: Prediction results.
        """
        # Simple prediction
        # For now, return dummy results
        # TODO: Implement proper prediction
        results = []
        for i in range(len(batch_img_metas)):
            result = InstanceData()
            result.scores = torch.softmax(cls_scores[0].flatten(), dim=0)
            results.append(result)
        return results


@MODELS.register_module()
class SimpleRegHead(BaseModule):
    """Simple regression head without bbox prediction.
    
    Args:
        in_channels (int): Number of input channels.
        out_dim (int): Output dimension.
        loss_weight (float): Loss weight.
        reg_type (str): Type of regression ('ph' or 'vbn').
    """
    
    def __init__(self,
                 in_channels: int,
                 out_dim: int = 1,
                 loss_weight: float = 1.0,
                 reg_type: str = 'ph',
                 init_cfg: ConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.loss_weight = loss_weight
        self.reg_type = reg_type  # 'ph' or 'vbn'
        
        # Simple regression conv
        self.reg_conv = nn.Conv2d(in_channels, out_dim, 1)
        
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """Forward function.
        
        Args:
            x (List[Tensor]): Features from FPN.
            
        Returns:
            List[Tensor]: Regression values.
        """
        reg_values = []
        for feat in x:
            reg_value = self.reg_conv(feat)
            reg_values.append(reg_value)
        return reg_values
    
    def loss_by_feat(self, reg_scores, batch_gt_instances, batch_img_metas):
        """Calculate regression loss for pH or VBN based on reg_type.
        
        Args:
            reg_scores (list[Tensor]): Regression scores for each FPN level
            batch_gt_instances: Ground truth instances (not used for image-level tasks)
            batch_img_metas (list[dict]): Meta information for each image
            
        Returns:
            dict: Loss dictionary with either 'ph_loss' or 'vbn_loss'
        """
        # Process each image in the batch
        total_loss = 0.0
        valid_samples = 0
        
        target_key = f'{self.reg_type}_value'  # 'ph_value' or 'vbn_value'
        
        for img_idx, img_meta in enumerate(batch_img_metas):
            target_value = img_meta.get(target_key, None)
            
            if target_value is None:
                continue
                
            # Get corresponding prediction for this image
            # reg_scores[0] has shape [batch_size, out_dim, H, W]
            # We need to pool it to get image-level prediction
            pred_for_img = reg_scores[0][img_idx]  # [out_dim, H, W]
            
            # Global average pooling to get image-level prediction
            pred_value = pred_for_img.mean(dim=[1, 2])  # [out_dim]
            
            # For regression, we expect out_dim=1, so take the first element
            if pred_value.dim() > 0:
                pred_value = pred_value[0]  # scalar
            
            # Convert target to tensor
            if not isinstance(target_value, torch.Tensor):
                target_tensor = torch.tensor(target_value, device=pred_value.device, dtype=torch.float32)
            else:
                target_tensor = target_value.to(device=pred_value.device, dtype=torch.float32)
            
            # Calculate MSE loss for this sample
            loss = F.mse_loss(pred_value, target_tensor)
            total_loss += loss
            valid_samples += 1
            
        if valid_samples == 0:
            # No valid samples, return zero loss
            device = reg_scores[0].device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            total_loss = total_loss / valid_samples
            
        loss_key = f'{self.reg_type}_loss'  # 'ph_loss' or 'vbn_loss'
        return {loss_key: total_loss * self.loss_weight}
    
    def predict_by_feat(self,
                       reg_values: List[Tensor],
                       batch_img_metas: List[dict],
                       rescale: bool = True) -> InstanceList:
        """Predict results.
        
        Args:
            reg_values (List[Tensor]): Regression values.
            batch_img_metas (List[dict]): Meta information.
            rescale (bool): Whether to rescale results.
            
        Returns:
            InstanceList: Prediction results.
        """
        # Simple prediction
        # For now, return dummy results
        # TODO: Implement proper prediction
        results = []
        for i in range(len(batch_img_metas)):
            result = InstanceData()
            result.scores = reg_values[0].flatten()
            results.append(result)
        return results 