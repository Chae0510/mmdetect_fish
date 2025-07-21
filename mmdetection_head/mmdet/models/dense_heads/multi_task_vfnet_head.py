# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads.vfnet_head import VFNetHead
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList


@MODELS.register_module()
class MultiTaskVFNetHead(VFNetHead):
    """Multi-task VFNet head with additional classification and regression heads.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of stacking convs of the head.
        feat_channels (int): Number of hidden channels.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Downsample
            factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_bbox_refine (:obj:`ConfigDict` or dict): Config of refined
            localization loss.
        loss_iou (:obj:`ConfigDict` or dict): Config of IoU loss.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Config dict for
            normalization layer. Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config of
            anchor-free head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor-free head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
        clf_heads (list[dict]): List of additional classification heads.
        ph_head (dict): Configuration of pH regression head.
        vbn_head (dict): Configuration of VBN regression head.
    """

    def __init__(self,
                 *args,
                 clf_heads: List[ConfigType] = None,
                 ph_head: ConfigType = None,
                 vbn_head: ConfigType = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Initialize additional classification heads
        if clf_heads is not None:
            self.clf_heads = nn.ModuleList()
            for i, head_cfg in enumerate(clf_heads):
                head = MODELS.build(head_cfg)
                # Set head index so each head knows which clf_score element to use
                head.head_idx = i
                self.clf_heads.append(head)
        else:
            self.clf_heads = None
            
        # Initialize regression heads
        if ph_head is not None:
            self.ph_head = MODELS.build(ph_head)
        else:
            self.ph_head = None
            
        if vbn_head is not None:
            self.vbn_head = MODELS.build(vbn_head)
        else:
            self.vbn_head = None

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        bbox_preds_refine: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        **kwargs) -> dict:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box energies / deltas
                for each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Get original VFNet losses
        losses = super().loss_by_feat(
            cls_scores, bbox_preds, bbox_preds_refine,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore,
            **kwargs)
        
        # Add additional classification losses
        if self.clf_heads is not None:
            clf_losses = self._clf_loss(cls_scores, batch_gt_instances,
                                       batch_img_metas)
            losses.update(clf_losses)
        
        # Add pH regression losses
        if self.ph_head is not None:
            ph_losses = self._ph_loss(cls_scores, batch_gt_instances,
                                     batch_img_metas)
            losses.update(ph_losses)
        
        # Add VBN regression losses
        if self.vbn_head is not None:
            vbn_losses = self._vbn_loss(cls_scores, batch_gt_instances,
                                       batch_img_metas)
            losses.update(vbn_losses)
        
        return losses

    def _clf_loss(self, cls_scores: List[Tensor],
                  batch_gt_instances: InstanceList,
                  batch_img_metas: List[dict]) -> dict:
        """Calculate additional classification losses.

        Args:
            cls_scores (list[Tensor]): Classification scores for each scale.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        clf_losses = dict()
        
        # Use cls_scores as feature maps (they have the right shape and are from FPN)
        # cls_scores shape: [batch_size, num_classes, H, W]
        # We need to create feature maps with 256 channels
        feature_maps = []
        for cls_score in cls_scores:
            # Create feature maps by repeating cls_score channels to get 256 channels
            batch_size, num_classes, H, W = cls_score.shape
            # Use the first few channels and repeat to get 256 channels
            if num_classes >= 256:
                feature_map = cls_score[:, :256, :, :]
            else:
                # Repeat the channels to get 256
                repeats = 256 // num_classes
                remainder = 256 % num_classes
                feature_map = cls_score.repeat(1, repeats, 1, 1)
                if remainder > 0:
                    feature_map = torch.cat([feature_map, cls_score[:, :remainder, :, :]], dim=1)
            feature_maps.append(feature_map)
        
        for i, head in enumerate(self.clf_heads):
            # Forward through the head with feature maps
            head_outputs = head(feature_maps)
            head_loss = head.loss_by_feat(head_outputs, batch_gt_instances, batch_img_metas)
            clf_losses[f'clf_{i}_loss'] = head_loss['clf_loss']
        
        return clf_losses

    def _ph_loss(self, cls_scores: List[Tensor],
                 batch_gt_instances: InstanceList,
                 batch_img_metas: List[dict]) -> dict:
        """Calculate pH regression losses.

        Args:
            cls_scores (list[Tensor]): Classification scores for each scale.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Create feature maps similar to _clf_loss
        feature_maps = []
        for cls_score in cls_scores:
            batch_size, num_classes, H, W = cls_score.shape
            if num_classes >= 256:
                feature_map = cls_score[:, :256, :, :]
            else:
                repeats = 256 // num_classes
                remainder = 256 % num_classes
                feature_map = cls_score.repeat(1, repeats, 1, 1)
                if remainder > 0:
                    feature_map = torch.cat([feature_map, cls_score[:, :remainder, :, :]], dim=1)
            feature_maps.append(feature_map)
        
        # Forward through the head with feature maps
        ph_outputs = self.ph_head(feature_maps)
        ph_loss = self.ph_head.loss_by_feat(ph_outputs, batch_gt_instances, batch_img_metas)
        return dict(ph_loss=ph_loss['ph_loss'])

    def _vbn_loss(self, cls_scores: List[Tensor],
                  batch_gt_instances: InstanceList,
                  batch_img_metas: List[dict]) -> dict:
        """Calculate VBN regression losses.

        Args:
            cls_scores (list[Tensor]): Classification scores for each scale.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Create feature maps similar to _clf_loss
        feature_maps = []
        for cls_score in cls_scores:
            batch_size, num_classes, H, W = cls_score.shape
            if num_classes >= 256:
                feature_map = cls_score[:, :256, :, :]
            else:
                repeats = 256 // num_classes
                remainder = 256 % num_classes
                feature_map = cls_score.repeat(1, repeats, 1, 1)
                if remainder > 0:
                    feature_map = torch.cat([feature_map, cls_score[:, :remainder, :, :]], dim=1)
            feature_maps.append(feature_map)
        
        # Forward through the head with feature maps
        vbn_outputs = self.vbn_head(feature_maps)
        vbn_loss = self.vbn_head.loss_by_feat(vbn_outputs, batch_gt_instances, batch_img_metas)
        return dict(vbn_loss=vbn_loss['vbn_loss'])

    def predict_by_feat(self,
                cls_scores: List[Tensor],
                bbox_preds: List[Tensor],
                bbox_preds_refine: List[Tensor] = None,
                batch_img_metas: List[dict] = None,
                rescale: bool = True,
                **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor], optional): Refined Box energies / deltas
                for each scale level, each is a 4D-tensor, the channel
                number is num_points * 4. Defaults to None.
            batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - clf_scores (list[Tensor]): Additional classification scores.
                - ph_values (Tensor): pH predictions.
                - vbn_values (Tensor): VBN predictions.
        """
        # Get original VFNet predictions
        if bbox_preds_refine is not None:
            # VFNet case with refinement
            results = super().predict_by_feat(
                cls_scores, bbox_preds, bbox_preds_refine,
                batch_img_metas, rescale, **kwargs)
        else:
            # Fallback: parent predict_by_feat expects (cls, bbox, img_metas, rescale)
            results = super().predict_by_feat(
                cls_scores, bbox_preds,
                batch_img_metas,  # positional third arg is img_metas
                rescale=rescale,  # keep as keyword for clarity
                **kwargs)
        
        # Create feature maps for additional heads (similar to loss calculation)
        feature_maps = []
        for cls_score in cls_scores:
            batch_size, num_classes, H, W = cls_score.shape
            if num_classes >= 256:
                feature_map = cls_score[:, :256, :, :]
            else:
                repeats = 256 // num_classes
                remainder = 256 % num_classes
                feature_map = cls_score.repeat(1, repeats, 1, 1)
                if remainder > 0:
                    feature_map = torch.cat([feature_map, cls_score[:, :remainder, :, :]], dim=1)
            feature_maps.append(feature_map)
        
        # Add additional predictions
        if self.clf_heads is not None:
            clf_results = []
            for head in self.clf_heads:
                # Forward through the head with feature maps
                head_outputs = head(feature_maps)
                clf_result = head.predict_by_feat(head_outputs, batch_img_metas, rescale)
                clf_results.append(clf_result)
            
            for i in range(len(results)):
                results[i].clf_scores = clf_results
        
        if self.ph_head is not None:
            # Forward through the head with feature maps
            ph_outputs = self.ph_head(feature_maps)
            ph_results = self.ph_head.predict_by_feat(ph_outputs, batch_img_metas, rescale)
            for i in range(len(results)):
                results[i].ph_values = ph_results[i].scores
        
        if self.vbn_head is not None:
            # Forward through the head with feature maps
            vbn_outputs = self.vbn_head(feature_maps)
            vbn_results = self.vbn_head.predict_by_feat(vbn_outputs, batch_img_metas, rescale)
            for i in range(len(results)):
                results[i].vbn_values = vbn_results[i].scores
        
        return results 