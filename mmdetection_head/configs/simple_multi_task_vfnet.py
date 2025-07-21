_base_ = '/workspace/mmdetect_fish/mmdetection/configs/vfnet/vfnet_r50_fpn_1x_coco.py'

# Dataset settings
dataset_type = 'CustomCocoDataset'
classes = ('whole_body', 'eye', 'gill')
data_root = '/workspace/fish_data/'

# Load pretrained VFNet checkpoint
load_from = '/workspace/mmdetect_fish/mmdetection/work_dirs/bcy_vfnet_r50_loss_cls_false/epoch_100.pth'

# Override model settings - MUCH SIMPLER!
model = dict(
    type='MultiTaskVFNet',  # Use our new detector
    backbone=dict(
        frozen_stages=4,  # Freeze all backbone stages
        norm_cfg=dict(type='BN', requires_grad=False),
    ),
    # bbox_head stays as original VFNetHead - no changes needed!
    bbox_head=dict(
        type='VFNetHead',  # Explicitly specify VFNetHead (not MultiTaskVFNetHead)
        num_classes=3,
    ),
    # Add simple image-level head
    image_head=dict(
        type='ImageLevelHead',
        in_channels=2048,  # ResNet50 final feature channels
        hidden_dim=512,
        num_clf_classes=8,  # 8 clf_score elements
        clf_num_classes=5,  # 1-5 range
        loss_weight=1.0
    )
)

# Dataset settings - same as before
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CustomCocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='subset_uniform/vbn_ph_uniform_train.json',
        data_prefix=dict(img='subset_uniform/train/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CustomCocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='subset_uniform/vbn_ph_uniform_val.json',
        data_prefix=dict(img='subset_uniform/val/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CustomCocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='subset_uniform/vbn_ph_uniform_test.json',
        data_prefix=dict(img='subset_uniform/test/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
            dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs')
        ]
    )
)

val_evaluator = dict(
    ann_file=data_root + 'subset_uniform/vbn_ph_uniform_val.json',
    metric='bbox'
)

test_evaluator = dict(
    ann_file=data_root + 'subset_uniform/vbn_ph_uniform_test.json',
    metric='bbox',
    classwise=True
)

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)

# Optimizer settings
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        bias_lr_mult=2.0,
        custom_keys=dict(
            backbone=dict(decay_mult=0.0, lr_mult=0.0),  # Freeze backbone
            neck=dict(decay_mult=0.0, lr_mult=0.0),      # Freeze neck
            bbox_head=dict(decay_mult=1.0, lr_mult=1.0), # Train bbox_head
            image_head=dict(decay_mult=1.0, lr_mult=1.0) # Train image_head
        ),
        norm_decay_mult=0.0),
    type='OptimWrapper')

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# Runtime settings
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False 