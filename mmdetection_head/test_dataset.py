#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mmdet.datasets import CustomCocoDataset
from mmengine.config import Config

def test_dataset():
    # Load config
    config = Config.fromfile('configs/multi_task_vfnet_fish.py')
    
    # Create dataset
    dataset = CustomCocoDataset(
        data_root='/workspace/fish_data/',
        ann_file='subset_uniform/vbn_ph_uniform_train.json',
        data_prefix=dict(img='subset_uniform/train/'),
        metainfo=dict(classes=('whole_body', 'eye', 'gill')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ]
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test first few samples
    for i in range(3):
        print(f"\n=== Sample {i} ===")
        data_info = dataset.get_data_info(i)
        print(f"Keys in data_info: {data_info.keys()}")
        
        if 'gt_instances' in data_info and data_info['gt_instances'] is not None:
            print(f"Number of gt_instances: {len(data_info['gt_instances'])}")
            if len(data_info['gt_instances']) > 0:
                first_gt = data_info['gt_instances'][0]
                print(f"First gt instance attributes: {dir(first_gt)}")
                
                # Check for our custom attributes
                if hasattr(first_gt, 'ph_value'):
                    print(f"ph_value: {first_gt.ph_value}")
                if hasattr(first_gt, 'vbn_value'):
                    print(f"vbn_value: {first_gt.vbn_value}")
                if hasattr(first_gt, 'clf_score'):
                    print(f"clf_score: {first_gt.clf_score}")

if __name__ == '__main__':
    test_dataset() 