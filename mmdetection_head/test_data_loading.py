#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspace/mmdetect_fish/mmdetection_head')

from mmdet.datasets import CustomCocoDataset
from mmengine.config import Config

# Test configuration
config = {
    'data_root': '/workspace/fish_data/',
    'ann_file': 'subset_uniform/vbn_ph_uniform_train.json',
    'data_prefix': dict(img='subset_uniform/train/'),
    'metainfo': dict(classes=('whole_body', 'eye', 'gill')),
    'pipeline': [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114)))
    ]
}

def test_data_loading():
    print("Testing CustomCocoDataset data loading...")
    
    # Create dataset
    dataset = CustomCocoDataset(**config)
    print(f"Dataset length: {len(dataset)}")
    
    # Test first few samples
    for i in range(min(5, len(dataset))):
        print(f"\n--- Sample {i} ---")
        data_info = dataset.get_data_info(i)
        
        print(f"Keys in data_info: {list(data_info.keys())}")
        
        if 'ph_value' in data_info:
            print(f"ph_value: {data_info['ph_value']}")
        else:
            print("ph_value: NOT FOUND")
            
        if 'vbn_value' in data_info:
            print(f"vbn_value: {data_info['vbn_value']}")
        else:
            print("vbn_value: NOT FOUND")
            
        if 'clf_score' in data_info:
            print(f"clf_score: {data_info['clf_score']}")
        else:
            print("clf_score: NOT FOUND")
        
        if 'gt_instances' in data_info and data_info['gt_instances']:
            print(f"Number of gt_instances: {len(data_info['gt_instances'])}")
            if len(data_info['gt_instances']) > 0:
                first_gt = data_info['gt_instances'][0]
                print(f"First gt_instances attributes: {dir(first_gt)}")
                
                if hasattr(first_gt, 'ph_value'):
                    print(f"First gt_instances.ph_value: {first_gt.ph_value}")
                if hasattr(first_gt, 'vbn_value'):
                    print(f"First gt_instances.vbn_value: {first_gt.vbn_value}")
                if hasattr(first_gt, 'clf_score'):
                    print(f"First gt_instances.clf_score: {first_gt.clf_score}")
        else:
            print("No gt_instances found")

if __name__ == "__main__":
    test_data_loading() 