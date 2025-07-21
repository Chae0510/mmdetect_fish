from typing import List, Tuple, Union

import torch
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, InstanceList
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class MultiTaskRoIHead(StandardRoIHead):
    """Multi-task ROI head with classification and regression heads.

    Args:
        bbox_roi_extractor (dict): Configuration of bbox roi extractor.
        bbox_head (dict): Configuration of bbox head.
        clf_heads (list[dict]): List of 9 classification heads configurations.
        ph_head (dict): Configuration of pH regression head.
        vbn_head (dict): Configuration of VBN regression head.
        mask_roi_extractor (dict, optional): Configuration of mask roi
            extractor. Defaults to None.
        mask_head (dict, optional): Configuration of mask head.
            Defaults to None.
        train_cfg (dict, optional): Configuration when training.
            Defaults to None.
        test_cfg (dict, optional): Configuration when testing.
            Defaults to None.
        init_cfg (dict, optional): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 bbox_roi_extractor: ConfigType,
                 bbox_head: ConfigType,
                 clf_heads: List[ConfigType],
                 ph_head: ConfigType,
                 vbn_head: ConfigType,
                 mask_roi_extractor: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        # Initialize 9 classification heads
        self.clf_heads = torch.nn.ModuleList([
            MODELS.build(head_cfg) for head_cfg in clf_heads
        ])
        
        # Initialize regression heads
        self.ph_head = MODELS.build(ph_head)
        self.vbn_head = MODELS.build(vbn_head)

    def loss(self, x: Tuple[torch.Tensor],
             rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the roi head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Get bbox losses
        bbox_losses = super().loss(x, rpn_results_list, batch_data_samples)
        
        # Get classification losses
        clf_losses = self._clf_loss(x, rpn_results_list, batch_data_samples)
        
        # Get pH and VBN regression losses
        ph_losses = self._ph_loss(x, rpn_results_list, batch_data_samples)
        vbn_losses = self._vbn_loss(x, rpn_results_list, batch_data_samples)
        
        # Combine all losses
        losses = dict()
        losses.update(bbox_losses)
        losses.update(clf_losses)
        losses.update(ph_losses)
        losses.update(vbn_losses)
        
        return losses

    def _clf_loss(self, x: Tuple[torch.Tensor],
                  rpn_results_list: InstanceList,
                  batch_data_samples: SampleList) -> dict:
        """Calculate classification losses for all 9 heads.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Extract features
        rois = self.bbox_roi_extractor(x, rpn_results_list)
        
        # Calculate losses for each classification head
        clf_losses = dict()
        for i, head in enumerate(self.clf_heads):
            clf_feats = head(rois)
            clf_targets = head.get_targets(rpn_results_list, batch_data_samples)
            clf_loss = head.loss(clf_feats, clf_targets)
            clf_losses[f'clf_{i}_loss'] = clf_loss
        
        return clf_losses

    def _ph_loss(self, x: Tuple[torch.Tensor],
                 rpn_results_list: InstanceList,
                 batch_data_samples: SampleList) -> dict:
        """Calculate pH regression loss.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Extract features
        rois = self.bbox_roi_extractor(x, rpn_results_list)
        ph_feats = self.ph_head(rois)
        
        # Get targets
        ph_targets = self.ph_head.get_targets(rpn_results_list,
                                            batch_data_samples)
        
        # Calculate loss
        ph_loss = self.ph_head.loss(ph_feats, ph_targets)
        
        return dict(ph_loss=ph_loss)

    def _vbn_loss(self, x: Tuple[torch.Tensor],
                  rpn_results_list: InstanceList,
                  batch_data_samples: SampleList) -> dict:
        """Calculate VBN regression loss.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # Extract features
        rois = self.bbox_roi_extractor(x, rpn_results_list)
        vbn_feats = self.vbn_head(rois)
        
        # Get targets
        vbn_targets = self.vbn_head.get_targets(rpn_results_list,
                                              batch_data_samples)
        
        # Calculate loss
        vbn_loss = self.vbn_head.loss(vbn_feats, vbn_targets)
        
        return dict(vbn_loss=vbn_loss)

    def predict(self,
                x: Tuple[torch.Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the roi head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (N, C, H, W).
            rpn_results_list (list[:obj:`InstanceData`]): list of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results to
                the original image. Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - clf_scores (list[Tensor]): List of classification scores
                    from 9 heads
                - ph_values (Tensor): pH predictions
                - vbn_values (Tensor): VBN predictions
        """
        # Get detection results
        results = super().predict(x, rpn_results_list, batch_data_samples, rescale)
        
        # Extract features
        rois = self.bbox_roi_extractor(x, rpn_results_list)
        
        # Get classification results
        clf_results = []
        for head in self.clf_heads:
            clf_feats = head(rois)
            clf_results.append(head.predict(clf_feats))
        
        # Get pH predictions
        ph_feats = self.ph_head(rois)
        ph_results = self.ph_head.predict(ph_feats)
        
        # Get VBN predictions
        vbn_feats = self.vbn_head(rois)
        vbn_results = self.vbn_head.predict(vbn_feats)
        
        # Combine results
        for i in range(len(results)):
            results[i].clf_scores = clf_results
            results[i].ph_values = ph_results[i]
            results[i].vbn_values = vbn_results[i]
        
        return results 