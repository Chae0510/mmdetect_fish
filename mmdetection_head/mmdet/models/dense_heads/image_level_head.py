# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List

from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import ConfigType


@MODELS.register_module()
class ImageLevelHead(BaseModule):
    """Simple image-level head for PH, VBN, and clf_score prediction.
    
    Takes backbone features and predicts image-level values.
    """
    
    def __init__(self,
                 in_channels: int = 2048,  # ResNet50 final feature
                 hidden_dim: int = 512,
                 num_clf_classes: int = 8,  # number of clf_score elements
                 clf_num_classes: int = 5,  # 1-5 range for each clf element
                 loss_weight: float = 1.0,
                 init_cfg: ConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_clf_classes = num_clf_classes
        self.clf_num_classes = clf_num_classes
        self.loss_weight = loss_weight
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # PH regression head (single value)
        self.ph_head = nn.Linear(hidden_dim, 1)
        
        # VBN regression head (single value)
        self.vbn_head = nn.Linear(hidden_dim, 1)
        
        # Clf score classification heads (8 separate classifiers)
        self.clf_heads = nn.ModuleList([
            nn.Linear(hidden_dim, clf_num_classes) 
            for _ in range(num_clf_classes)
        ])
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward function.
        
        Args:
            x (Tensor): Backbone feature map [B, C, H, W]
            
        Returns:
            Dict[str, Tensor]: Predictions
        """
        # Global pooling: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        feat = self.global_pool(x).flatten(1)
        
        # Shared feature
        shared_feat = self.shared_fc(feat)  # [B, hidden_dim]
        
        # Predictions
        ph_pred = self.ph_head(shared_feat)  # [B, 1]
        vbn_pred = self.vbn_head(shared_feat)  # [B, 1]
        
        # Clf predictions
        clf_preds = []
        for clf_head in self.clf_heads:
            clf_pred = clf_head(shared_feat)  # [B, 5]
            clf_preds.append(clf_pred)
        
        return {
            'ph_pred': ph_pred,
            'vbn_pred': vbn_pred,
            'clf_preds': clf_preds  # List of [B, 5] tensors
        }
    
    def loss(self, 
             predictions: Dict[str, Tensor],
             batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Calculate loss.
        
        Args:
            predictions: Model predictions
            batch_img_metas: Image metadata containing gt values
            
        Returns:
            Dict[str, Tensor]: Loss components
        """
        losses = {}
        
        # Extract ground truth values from batch_img_metas
        ph_gts = []
        vbn_gts = []
        clf_gts = []
        
        for img_meta in batch_img_metas:
            # Get PH, VBN, clf_score from img_meta
            ph_gts.append(img_meta.get('ph_value', 0.0))
            vbn_gts.append(img_meta.get('vbn_value', 0.0))
            clf_gts.append(img_meta.get('clf_score', [3] * self.num_clf_classes))
        
        device = predictions['ph_pred'].device
        
        # PH loss
        if ph_gts:
            ph_gt_tensor = torch.tensor(ph_gts, device=device, dtype=torch.float32).unsqueeze(1)
            losses['ph_loss'] = self.mse_loss(predictions['ph_pred'], ph_gt_tensor) * self.loss_weight
        
        # VBN loss
        if vbn_gts:
            vbn_gt_tensor = torch.tensor(vbn_gts, device=device, dtype=torch.float32).unsqueeze(1)
            losses['vbn_loss'] = self.mse_loss(predictions['vbn_pred'], vbn_gt_tensor) * self.loss_weight
        
        # Clf losses (each element separately)
        if clf_gts:
            clf_losses = []
            for i in range(self.num_clf_classes):
                clf_gt_list = [clf_gt[i] - 1 for clf_gt in clf_gts]  # Convert 1-5 to 0-4
                clf_gt_tensor = torch.tensor(clf_gt_list, device=device, dtype=torch.long)
                clf_loss = self.ce_loss(predictions['clf_preds'][i], clf_gt_tensor)
                clf_losses.append(clf_loss)
                losses[f'clf_{i}_loss'] = clf_loss * self.loss_weight
            
            # Total clf loss
            losses['clf_total_loss'] = sum(clf_losses) * self.loss_weight
        
        return losses 