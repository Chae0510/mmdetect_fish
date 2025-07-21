_base_ = [
    '../_base_/models/multi_task_rcnn_r50_fpn.py',  # Multi-task model config
    '../_base_/datasets/coco_detection.py',         # Dataset config
    '../_base_/schedules/schedule_1x.py',           # Training schedule
    '../_base_/default_runtime.py'                  # Runtime settings
]

# Dataset settings
data_root = 'data/fish/'  # Change this to your dataset path
dataset_type = 'CocoDataset'

# Model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),  # Detection classes (fish)
        cls_head=dict(num_classes=3),   # Classification classes (e.g., species)
        reg_head=dict(num_classes=1)    # Regression output (e.g., size/weight)
    )
)

# Training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Dataset pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Data loader settings
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

# Runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Visualization settings
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer') 