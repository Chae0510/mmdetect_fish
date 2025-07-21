_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', 
    '_base_/default_runtime.py'
]

# Custom dataset
dataset_type = 'CustomCocoDataset'
data_root = '/workspace/fish_data/wholebody_subset/'

# Model configuration
model = dict(
    type='BboxLevelVFNet',  # Use BboxLevelVFNet instead
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='BboxLevelVFNetHead',  # Use BboxLevelVFNetHead
        num_classes=3,  # whole_body, eye, gill
        in_channels=256,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        # VFNet configs
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        ph_loss_weight=0.5,
        vbn_loss_weight=0.2,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# Dataset configuration without WholeBodyCrop for debugging
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='WholeBodyCrop', padding_ratio=0.2, min_crop_size=(256, 256)),
    dict(type='NormalizeAttr', ph_div=14.0, vbn_div=50.0),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', 
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'scale_factor', 'flip', 'flip_direction',
                   'ph_value', 'vbn_value', 'clf_score'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='WholeBodyCrop', padding_ratio=0.2, min_crop_size=(256, 256)),
    dict(type='NormalizeAttr', ph_div=14.0, vbn_div=50.0),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'ph_value', 'vbn_value', 'clf_score'))
]

train_dataloader = dict(
    batch_size=24,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wholebody_bbox_train_clean.json',
        data_prefix=dict(img='train/'),
        filter_cfg=None,  # Explicitly set to None to disable all filters
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wholebody_bbox_val_clean.json',
        data_prefix=dict(img='val/'),  # Fixed image path
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wholebody_bbox_test_clean.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

# Evaluation configuration
val_evaluator = [
    dict(type='AttrMetric'),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wholebody_bbox_val_clean.json',
        metric='bbox',
        format_only=False,
        backend_args=None)
]

test_evaluator = [
    dict(type='AttrMetric'),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wholebody_bbox_test_clean.json',
        metric='bbox',
        format_only=False,
        backend_args=None)
]

# Optimizer configuration (AMP for speed & memory)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2))

# Learning rate configuration
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=36, by_epoch=True, eta_min=1e-6)
]

# Training configuration
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# Logging
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Ensure custom metric class is discoverable
custom_imports = dict(
    imports=['mmdet.evaluation.metrics.attr_metric'],
    allow_failed_imports=False)

# Load pretrained weights
load_from = None
resume = False

# Automatically scale learning-rate according to the new batch size
auto_scale_lr = dict(enable=True, base_batch_size=8) 