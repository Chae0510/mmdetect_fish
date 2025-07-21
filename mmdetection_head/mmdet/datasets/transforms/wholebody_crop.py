# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData


@TRANSFORMS.register_module()
class WholeBodyCrop:
    """Crop image around whole body bbox with padding.
    
    This transform finds the whole body bbox and crops the image around it
    with additional padding. All other bboxes (eye, gill) are adjusted
    to the new coordinate system.
    
    Args:
        padding_ratio (float): Ratio of padding relative to whole body bbox size.
            Default: 0.2 (20% padding)
        min_crop_size (tuple): Minimum crop size (width, height). 
            Default: (256, 256)
    """
    
    def __init__(self, 
                 padding_ratio: float = 0.2,
                 min_crop_size: Tuple[int, int] = (256, 256)):
        self.padding_ratio = padding_ratio
        self.min_crop_size = min_crop_size
        self.debug_count = 0
    
    def __call__(self, results: Dict) -> Dict:
        """Call function to apply the transform."""
        return self.transform(results)
    
    def transform(self, results: Dict) -> Dict:
        """Transform function to crop around whole body."""
        
        self.debug_count += 1
        if self.debug_count <= 5:  # Debug first 5 samples
            print(f"\n=== WholeBodyCrop Debug {self.debug_count} ===")
        
        # Get image and annotations
        img = results['img']
        h, w = img.shape[:2]
        
        # Get bboxes and labels; gracefully skip if missing (e.g. empty-GT image)
        if 'gt_bboxes' not in results or 'gt_labels' not in results:
            if self.debug_count <= 5:
                print('WholeBodyCrop: missing gt_bboxes or gt_labels, skip')
            return results
        
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']
        
        if self.debug_count <= 5:
            print(f"Image shape: {img.shape}")
            print(f"GT labels: {gt_labels}")
            print(f"GT bboxes shape: {gt_bboxes.shape if hasattr(gt_bboxes, 'shape') else 'No shape attr'}")
            print(f"GT bboxes type: {type(gt_bboxes)}")
            if hasattr(gt_bboxes, 'tensor'):
                print(f"Bboxes: {gt_bboxes.tensor}")
            else:
                print(f"Bboxes: {gt_bboxes}")
        
        # Find whole body bbox (category_id = 1, but in gt_labels it's index 0)
        # Assuming labels are 0-indexed: whole_body=0, eye=1, gill=2
        whole_body_indices = np.where(gt_labels == 0)[0]  # whole_body class index
        
        if self.debug_count <= 5:
            print(f"Whole body indices: {whole_body_indices}")
        
        if len(whole_body_indices) == 0:
            # No whole body found, return original
            if self.debug_count <= 5:
                print("Warning: No whole_body bbox found, skipping crop")
            return results
        
        # Use first whole body bbox
        wb_idx = whole_body_indices[0]
        wb_bbox = gt_bboxes[wb_idx]  # [x1, y1, x2, y2]
        
        if self.debug_count <= 5:
            print(f"Selected whole body bbox: {wb_bbox}")
        
        # Convert to [x, y, w, h] format
        x1, y1, x2, y2 = wb_bbox
        wb_x, wb_y = x1, y1
        wb_w, wb_h = x2 - x1, y2 - y1
        
        # Calculate padding
        pad_w = int(wb_w * self.padding_ratio)
        pad_h = int(wb_h * self.padding_ratio)
        
        # Calculate crop coordinates
        crop_x1 = max(0, int(wb_x - pad_w))
        crop_y1 = max(0, int(wb_y - pad_h))
        crop_x2 = min(w, int(wb_x + wb_w + pad_w))
        crop_y2 = min(h, int(wb_y + wb_h + pad_h))
        
        # Ensure minimum crop size
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        if crop_w < self.min_crop_size[0]:
            expand_w = (self.min_crop_size[0] - crop_w) // 2
            crop_x1 = max(0, crop_x1 - expand_w)
            crop_x2 = min(w, crop_x1 + self.min_crop_size[0])
            
        if crop_h < self.min_crop_size[1]:
            expand_h = (self.min_crop_size[1] - crop_h) // 2
            crop_y1 = max(0, crop_y1 - expand_h)
            crop_y2 = min(h, crop_y1 + self.min_crop_size[1])
        
        # Crop image
        cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Adjust all bboxes to new coordinate system
        adjusted_bboxes = gt_bboxes.clone()
        adjusted_bboxes[:, [0, 2]] -= crop_x1  # x1, x2
        adjusted_bboxes[:, [1, 3]] -= crop_y1  # y1, y2
        
        # Clip bboxes to crop boundaries
        crop_w_new = crop_x2 - crop_x1
        crop_h_new = crop_y2 - crop_y1
        
        adjusted_bboxes[:, 0] = np.clip(adjusted_bboxes[:, 0], 0, crop_w_new)  # x1
        adjusted_bboxes[:, 1] = np.clip(adjusted_bboxes[:, 1], 0, crop_h_new)  # y1
        adjusted_bboxes[:, 2] = np.clip(adjusted_bboxes[:, 2], 0, crop_w_new)  # x2
        adjusted_bboxes[:, 3] = np.clip(adjusted_bboxes[:, 3], 0, crop_h_new)  # y2
        
        # Filter out invalid bboxes (too small after clipping)
        bbox_w = adjusted_bboxes[:, 2] - adjusted_bboxes[:, 0]
        bbox_h = adjusted_bboxes[:, 3] - adjusted_bboxes[:, 1]
        valid_mask = (bbox_w > 1) & (bbox_h > 1)
        
        if self.debug_count <= 5:
            print(f"Crop coordinates: ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})")
            print(f"Cropped image shape: {cropped_img.shape}")
            print(f"Valid bbox mask: {valid_mask}")
            print(f"Number of valid bboxes: {valid_mask.sum()}")
        
        # Update results
        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape[:2]
        results['gt_bboxes'] = adjusted_bboxes[valid_mask]
        results['gt_labels'] = gt_labels[valid_mask]
        
        # Update other annotations if they exist
        if 'gt_instances' in results:
            gt_instances = results['gt_instances']
            gt_instances.bboxes = adjusted_bboxes[valid_mask]
            gt_instances.labels = gt_labels[valid_mask]
            
            # Update other attributes if they exist
            for attr in ['ph_value', 'vbn_value', 'clf_score']:
                if hasattr(gt_instances, attr):
                    attr_values = getattr(gt_instances, attr)
                    setattr(gt_instances, attr, attr_values[valid_mask])
        
        # Store crop info for debugging
        results['crop_info'] = {
            'original_size': (w, h),
            'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2),
            'whole_body_bbox': [wb_x, wb_y, wb_w, wb_h],
            'padding_ratio': self.padding_ratio
        }
        
        return results
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'padding_ratio={self.padding_ratio}, '
                f'min_crop_size={self.min_crop_size})') 