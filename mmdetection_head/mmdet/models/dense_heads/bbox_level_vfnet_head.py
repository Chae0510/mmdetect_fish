# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.dense_heads.vfnet_head import VFNetHead
import numpy as np
import datetime, os
import matplotlib.pyplot as plt
import seaborn as sns
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures import DetDataSample
from mmdet.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class BboxLevelVFNetHead(VFNetHead):
    """VFNet head with bbox-level attribute prediction.
    
    Extends VFNetHead to predict ph, vbn, and clf_score for each 
    detected bounding box.
    """
    
    def __init__(self, 
                 num_classes: int,
                 in_channels: int,
                 ph_loss_weight: float = 0.05,
                 vbn_loss_weight: float = 0.02,
                 clf_loss_weight: float = 1.0,
                 clf_last_loss_weight: float | None = None,
                 last_head_class_weights: Optional[List[float]] = None,
                 **kwargs):
        """Args:
        ph_loss_weight (float): weight applied to pH MSE loss.
        vbn_loss_weight (float): weight applied to VBN MSE loss.
        """

        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)
        
        # Add attribute prediction heads
        self.ph_head = nn.Linear(in_channels, 1)
        self.vbn_head = nn.Linear(in_channels, 1)
        # Create 8 separate classification heads (each predicts 5-class logits)
        self.clf_heads = nn.ModuleList([
            nn.Linear(in_channels, 5) for _ in range(8)
        ])

        # Store loss weights
        self.ph_w = ph_loss_weight
        self.vbn_w = vbn_loss_weight
        self.clf_w = clf_loss_weight
        self.clf_last_w = clf_last_loss_weight if clf_last_loss_weight is not None else clf_loss_weight
        # Expect list of 5 floats corresponding to classes 1..5; normalize inside loss
        self.last_head_cls_weights = last_head_class_weights

        # Confusion matrix for training (GT rows, Pred cols) for last head
        self.register_buffer('train_confmat', torch.zeros((5, 5), dtype=torch.int64), persistent=False)
    
    def forward(self, feats: Tuple[Tensor]) -> Tuple:
        """Forward pass of the head."""
        # Standard VFNet forward pass
        parent_output = super().forward(feats)  # type: ignore[arg-type]
        
        # VFNetHead returns (cls_scores, bbox_preds, bbox_preds_refine) during
        # training and (cls_scores, bbox_preds_refine) during inference.
        if self.training:
            cls_scores, bbox_preds, bbox_preds_refine = parent_output  # type: ignore[misc]
        else:
            # Inference / validation mode. Follow the signature expected by
            # BaseDenseHead.predict, which relies on forward() returning
            # ONLY the detection outputs that will be fed into
            # `predict_by_feat`. Returning extra tensors here will break the
            # unpacking logic inside `BaseDenseHead.predict`.

            cls_scores, bbox_preds_refine = parent_output  # type: ignore[misc]

            # --- Attribute predictions (optional) ---
            # We still compute attribute predictions so that custom
            # post-processing (outside of the standard MMDet evaluation
            # pipeline) can make use of them if desired. However, we will *not*
            # include them in the return value to keep compatibility with
            # MMDet's built-in inference flow.

            _ = []  # placeholders for attribute outputs (not returned)
            for i, feat in enumerate(feats):
                cls_feat = feat
                for cls_conv in self.cls_convs:
                    cls_feat = cls_conv(cls_feat)
                B, C, H, W = cls_feat.shape
                pooled_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1)).view(B, C)
                _.append(self.ph_head(pooled_feat))  # compute once to keep params trained

            # NOTE: We deliberately return only (cls_scores, bbox_preds_refine)
            # because `BaseDenseHead.predict` expects exactly two (or three)
            # elements and will forward-unpack them into `predict_by_feat`.

            return cls_scores, bbox_preds_refine
        
        # Extract attribute predictions from the classification features
        ph_preds = []
        vbn_preds = []
        clf_preds = []
        
        for i, feat in enumerate(feats):
            # Use the same feature processing as VFNetHead for consistency
            cls_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            
            # Global average pooling over spatial dimensions for attribute prediction
            B, C, H, W = cls_feat.shape
            pooled_feat = F.adaptive_avg_pool2d(cls_feat, (1, 1)).view(B, C)
            
            ph_preds.append(torch.sigmoid(self.ph_head(pooled_feat)))  # 0-1
            vbn_preds.append(torch.sigmoid(self.vbn_head(pooled_feat)))  # 0-1
            # Collect predictions from all 8 heads and stack to shape [B, 8, 5]
            head_outputs = [head(pooled_feat) for head in self.clf_heads]  # list of [B,5]
            clf_pred_level = torch.stack(head_outputs, dim=1)  # [B, 8, 5]
            clf_preds.append(clf_pred_level)
        
        if bbox_preds_refine is not None:
            return cls_scores, bbox_preds, bbox_preds_refine, ph_preds, vbn_preds, clf_preds
        else:
            return cls_scores, bbox_preds, ph_preds, vbn_preds, clf_preds
    
    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     bbox_preds_refine: List[Tensor],
                     ph_preds: List[Tensor],
                     vbn_preds: List[Tensor], 
                     clf_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """Compute losses of the head.
        
        Args:
            cls_scores: Classification scores for all scale levels.
            bbox_preds: Box energies / deltas for all scale levels.
            bbox_preds_refine: Refined box energies / deltas for all scale levels.
            ph_preds: PH predictions for all scale levels.
            vbn_preds: VBN predictions for all scale levels.
            clf_preds: CLF score predictions for all scale levels.
            batch_gt_instances: Ground truth instances for each image.
            batch_img_metas: List of image meta information.
            batch_gt_instances_ignore: Ground truth instances to be ignored.
            
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Use parent class loss_by_feat for detection losses  
        detection_losses = super().loss_by_feat(
            cls_scores, bbox_preds, bbox_preds_refine,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore
        )
        
        # ---------------- Attribute losses ----------------
        attr_ph_loss  = 0.0
        attr_vbn_loss = 0.0
        attr_clf_loss_heads = [torch.tensor(0.0, device=bbox_preds[0].device) for _ in range(8)]  # per-head accumulators
        valid_img_cnt = 0

        num_levels = len(ph_preds)

        # ---------- TEMP DEBUG START ----------
        if not hasattr(self, '_dbg_cnt'):
            self._dbg_cnt = 0  # type: ignore[attr-defined]

        if self._dbg_cnt < 5:  # print first 5 calls only
            print("\n[DEBUG] BboxLevelVFNetHead.loss_by_feat call", self._dbg_cnt)
            for img_idx, (gt_inst, meta) in enumerate(zip(batch_gt_instances, batch_img_metas)):
                num_gts = len(gt_inst.bboxes) if hasattr(gt_inst, 'bboxes') else 0
                print(f"  Img {img_idx}: GTs={num_gts}, ph={meta.get('ph_value')}, vbn={meta.get('vbn_value')}, clf_len={len(meta.get('clf_score', [])) if meta.get('clf_score') is not None else 'None'}")

        self._dbg_cnt += 1  # type: ignore[attr-defined]
        # ---------- TEMP DEBUG END ----------

        # Iterate over batch images
        batch_size = len(batch_img_metas)
        for img_idx in range(batch_size):
            meta = batch_img_metas[img_idx]

            # 1. pH, VBN regression targets (scalar)
            ph_tgt  = meta.get('ph_value',  None)
            vbn_tgt = meta.get('vbn_value', None)
            clf_tgt = meta.get('clf_score', None)  # expected list length 8

            if ph_tgt is None or vbn_tgt is None or clf_tgt is None:
                continue  # skip if any target missing

            valid_img_cnt += 1

            # Average predictions over FPN levels for this image
            ph_pred_img  = torch.stack([lvl_ph[img_idx]  for lvl_ph  in ph_preds]).mean()   # normalized 0-1
            vbn_pred_img = torch.stack([lvl_vbn[img_idx] for lvl_vbn in vbn_preds]).mean()  # normalized 0-1

            # Convert targets to tensors on same device
            device = ph_pred_img.device
            ph_target_tensor  = torch.tensor(float(ph_tgt) / 14.0,  device=device)
            vbn_target_tensor = torch.tensor(float(vbn_tgt) / 50.0, device=device)

            # Use float32 for stability, and apply weights later
            attr_ph_loss  += F.mse_loss(ph_pred_img.float(),  ph_target_tensor.float(),  reduction='mean')
            attr_vbn_loss += F.mse_loss(vbn_pred_img.float(), vbn_target_tensor.float(), reduction='mean')

            # ---------- clf heads (8 heads × 5-class CE) ----------
            clf_pred_lvls = [lvl_clf[img_idx] for lvl_clf in clf_preds]  # list len L each [8,5]
            clf_pred_stack = torch.stack(clf_pred_lvls, dim=0).mean(dim=0)  # [8,5]

            for head_idx in range(8):
                logits = clf_pred_stack[head_idx]  # [5]
                target_val = int(clf_tgt[head_idx])
                # target scale 1~5 → 0~4
                target_class = torch.tensor(target_val - 1, device=device).clamp(0, 4)
                if head_idx == 7 and self.last_head_cls_weights is not None:
                    w_tensor = torch.tensor(self.last_head_cls_weights, device=device, dtype=logits.dtype)
                    # Ensure length 5; if not, fallback to uniform
                    if w_tensor.numel() != 5:
                        w_tensor = None
                    loss_val = F.cross_entropy(logits.unsqueeze(0), target_class.unsqueeze(0), weight=w_tensor)
                else:
                    loss_val = F.cross_entropy(logits.unsqueeze(0), target_class.unsqueeze(0))

                attr_clf_loss_heads[head_idx] += loss_val

                # ------------- accumulate train confusion (only last head) -------------
                if head_idx == 7 and (0 <= target_class.item() < 5) and (0 <= logits.argmax().item() < 5):
                    gt_i = int(target_class.item())
                    pred_i = int(logits.argmax().item())
                    self.train_confmat[gt_i, pred_i] += 1

        if valid_img_cnt > 0:
            attr_ph_loss  = attr_ph_loss  / valid_img_cnt
            attr_vbn_loss = attr_vbn_loss / valid_img_cnt
            attr_clf_loss_heads = [l / valid_img_cnt for l in attr_clf_loss_heads]
        else:
            attr_ph_loss  = torch.tensor(0.0, device=bbox_preds[0].device, requires_grad=True)
            attr_vbn_loss = torch.tensor(0.0, device=bbox_preds[0].device, requires_grad=True)
            attr_clf_loss_heads = [torch.tensor(0.0, device=bbox_preds[0].device, requires_grad=True) for _ in range(8)]

        # Apply weights
        detection_losses['loss_ph']  = attr_ph_loss  * self.ph_w
        detection_losses['loss_vbn'] = attr_vbn_loss * self.vbn_w

        # log per-head classification losses
        # Per-head scaling (last head can have different weight)
        scaled_losses = []
        for h_idx, h_loss in enumerate(attr_clf_loss_heads):
            w = self.clf_last_w if h_idx == 7 else self.clf_w
            scaled = h_loss * w
            detection_losses[f'loss_clf_h{h_idx}'] = scaled
            scaled_losses.append(scaled)

        # overall mean classification loss (weighted)
        detection_losses['loss_clf'] = sum(scaled_losses) / 8
        
        return detection_losses 

    # ------------------------------------------------------------------
    # Utility to save accumulated training confusion matrix after epoch
    # ------------------------------------------------------------------
    def save_train_confmat(self, logger=None, save_dir: str | None = None):
        """Dump and reset the accumulated training confusion matrix.

        Args:
            logger: MMEngine logger instance used to log the save path.
            save_dir: Explicit directory to save files. If ``None``, the
                function will try to infer the directory from the logger's
                first file handler. If that fails, it falls back to the
                current working directory.
        """

        if self.train_confmat.sum() == 0:
            return  # nothing accumulated

        # Resolve save directory priority: explicit > logger > cwd
        final_dir: str | None = save_dir

        if final_dir is None and logger is not None and hasattr(logger, 'handlers'):
            for h in logger.handlers:  # type: ignore[attr-defined]
                if hasattr(h, 'baseFilename'):
                    final_dir = os.path.dirname(h.baseFilename)
                    break

        if final_dir is None:
            final_dir = os.getcwd()

        # Make sure it exists
        os.makedirs(final_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime('%m%d%H%M%S')
        npy_path = os.path.join(final_dir, f'confmat_train_head_{ts}.npy')
        np.save(npy_path, self.train_confmat.cpu().numpy())

        # png
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(self.train_confmat.cpu().numpy(), annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=[f'P{i}' for i in range(1,6)], yticklabels=[f'T{i}' for i in range(1,6)])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Ground Truth')
            ax.set_title('Train Confusion Matrix')
            png_path = os.path.join(final_dir, f'confmat_train_head_{ts}.png')
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close(fig)
            if logger:
                logger.info(f'Saved train confusion matrix to {npy_path} and {png_path}')
        except Exception as e:
            if logger:
                logger.info(f'Failed to save train confusion matrix image: {e}. NPY at {npy_path}')

        # reset
        self.train_confmat.zero_()

    # ------------------------------------------------------------------
    # Inference helper: attach attribute predictions to InstanceData so that
    # custom metrics can evaluate them. We keep detection pipeline intact by
    # delegating bbox logic to BaseDenseHead.predict(), then simply augment
    # its outputs.
    # ------------------------------------------------------------------
    def predict(self, x: Tuple[Tensor], batch_data_samples: List["DetDataSample"], rescale: bool = False):  # type: ignore[override]
        """Override BaseDenseHead.predict.

        Returns a list[InstanceData] with extra fields:
            - ph_pred  (Tensor, shape ())
            - vbn_pred (Tensor, shape ())
            - clf_logits (Tensor, shape (8,5))
        """

        # First, obtain standard detection results via the parent method.
        det_results = super().predict(x, batch_data_samples, rescale=rescale)

        # ---- Compute attribute predictions per image ----
        ph_lvls: List[Tensor] = []
        vbn_lvls: List[Tensor] = []
        clf_lvls: List[Tensor] = []  # each [B,8,5]

        for feat in x:
            cls_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)

            B, C, H, W = cls_feat.shape
            pooled = F.adaptive_avg_pool2d(cls_feat, (1, 1)).view(B, C)

            ph_lvls.append(torch.sigmoid(self.ph_head(pooled)))        # [B,1] normalized
            vbn_lvls.append(torch.sigmoid(self.vbn_head(pooled)))      # [B,1] normalized
            clf_lvls.append(torch.stack([h(pooled) for h in self.clf_heads], dim=1))  # [B,8,5]

        # average over FPN levels
        ph_batch_norm  = torch.stack(ph_lvls, dim=0).mean(dim=0).squeeze(-1)    # [B]
        vbn_batch_norm = torch.stack(vbn_lvls, dim=0).mean(dim=0).squeeze(-1)   # [B]
        clf_batch = torch.stack(clf_lvls, dim=0).mean(dim=0)                     # [B,8,5]

        # Convert back to raw scale for evaluation/storage
        ph_batch_raw  = ph_batch_norm * 14.0
        vbn_batch_raw = vbn_batch_norm * 50.0

        # attach to each InstanceData in det_results
        for idx, inst in enumerate(det_results):
            num_det = len(inst.bboxes) if hasattr(inst, 'bboxes') else len(inst)

            # broadcast scalar predictions to match detection count
            if num_det > 0:
                inst.ph_pred = ph_batch_raw[idx].repeat(num_det)
                inst.vbn_pred = vbn_batch_raw[idx].repeat(num_det)
                # clf_logits: repeat along 0 dim to [num_det,8,5]
                inst.clf_logits = clf_batch[idx].unsqueeze(0).repeat(num_det, 1, 1)
            else:
                device = ph_batch_raw.device if isinstance(ph_batch_raw, torch.Tensor) else 'cpu'
                inst.ph_pred = ph_batch_raw[idx].new_empty((0,))
                inst.vbn_pred = vbn_batch_raw[idx].new_empty((0,))
                inst.clf_logits = clf_batch[idx].new_empty((0, 8, 5))

        return det_results 