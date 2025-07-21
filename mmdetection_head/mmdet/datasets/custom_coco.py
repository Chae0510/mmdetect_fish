# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Dict, Any
import copy
from mmengine.fileio import get_local_path
from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    """Custom COCO dataset that preserves self.coco and adds bbox-level attributes.
    
    This dataset:
    1. Keeps self.coco available at runtime (doesn't delete it after loading)
    2. Adds bbox-level attributes (ph_value, vbn_value, clf_score) to data_info
    """

    # Override metainfo so that COCOAPI correctly maps category ids (1,2,3)
    # to the names used in our JSON files. Without this, all annotations
    # are filtered out because their category names are not found in the
    # default 80-class COCO list.
    METAINFO = {
        'classes': ('whole_body', 'eye', 'gill'),
        # simple color palette for visualization
        'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    }

    def load_data_list(self) -> List[dict]:
        """Load data list and preserve self.coco for runtime access."""
        # Call parent's load_data_list which handles all the complex logic
        data_list = super().load_data_list()
        
        # The parent method deletes self.coco, so we need to reload it
        with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        
        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index and add custom bbox-level attributes."""
        # Get standard data info from parent
        data_info = super().get_data_info(idx)
        
        # Add custom attributes if self.coco is available
        if hasattr(self, 'coco') and self.coco is not None:
            self._add_bbox_attributes(data_info)
        
        return data_info

    def _add_bbox_attributes(self, data_info: dict) -> None:
        """Add bbox-level attributes to data_info."""
        try:
            img_id = data_info['img_id']
            
            # Get all annotations for this image
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            anns = self.coco.load_anns(ann_ids)
            
            # Debug: Check what we found
            if not anns:
                return  # No annotations for this image
            
            # Look for any annotation that has our custom attributes
            found_attrs = False
            for ann in anns:
                # Check if this annotation has any of our custom attributes
                has_ph = 'ph' in ann
                has_vbn = 'vbn' in ann  
                has_clf = 'clf_score' in ann
                
                if has_ph or has_vbn or has_clf:
                    # Add whatever attributes we find
                    if has_ph:
                        data_info['ph_value'] = float(ann['ph'])
                    if has_vbn:
                        data_info['vbn_value'] = float(ann['vbn'])
                    if has_clf:
                        data_info['clf_score'] = ann['clf_score']
                    found_attrs = True
                    break
                    
            # If we didn't find the expected attributes, check what keys are available
            if not found_attrs and anns:
                # Check all available keys in first annotation
                available_keys = list(anns[0].keys())
                # Look for similar keys that might contain our data
                for key in available_keys:
                    if 'ph' in key.lower():
                        data_info['ph_value'] = float(anns[0][key])
                        found_attrs = True
                    elif 'vbn' in key.lower():
                        data_info['vbn_value'] = float(anns[0][key])
                        found_attrs = True
                    elif 'clf' in key.lower() or 'score' in key.lower():
                        data_info['clf_score'] = anns[0][key]
                        found_attrs = True
                        
        except Exception as e:
            # If anything goes wrong, just continue without custom attributes
            pass

    def __getstate__(self):
        """Handle pickling by preserving self.coco."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Handle unpickling by restoring self.coco."""
        self.__dict__.update(state)
        # Reload self.coco if it's missing
        if not hasattr(self, 'coco') or self.coco is None:
            try:
                with get_local_path(self.ann_file, backend_args=self.backend_args) as local_path:
                    self.coco = self.COCOAPI(local_path)
            except Exception:
                pass 