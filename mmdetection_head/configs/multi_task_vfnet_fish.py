_base_ = '/workspace/mmdetect_fish/mmdetection/configs/vfnet/vfnet_r50_fpn_1x_coco.py'

# Dataset settings
dataset_type = 'CustomCocoDataset'
classes = ('whole_body', 'eye', 'gill')
data_root = '/workspace/fish_data/'

# Define categories for the dataset
categories = [
    {'id': 1, 'name': 'whole_body'},
    {'id': 2, 'name': 'eye'},
    {'id': 3, 'name': 'gill'}
]

# Load pretrained VFNet checkpoint
load_from = '/workspace/mmdetect_fish/mmdetection/work_dirs/bcy_vfnet_r50_loss_cls_false/epoch_100.pth'

# Override model settings for multi-task
model = dict(
    backbone=dict(
        frozen_stages=4,  # Freeze all backbone stages
        norm_cfg=dict(type='BN', requires_grad=False),
    ),
    neck=dict(
        # frozen=True 제거 - FPN에서 지원하지 않음
    ),
    bbox_head=dict(
        type='MultiTaskVFNetHead',
        num_classes=3,
        # Simplified additional heads - just basic classification heads
        clf_heads=[
            dict(
                type='SimpleClfHead',
                in_channels=256,
                num_classes=5,  # 1-5 range prediction
                loss_weight=0.1)
            for _ in range(8)  # 8 heads
        ],
        # pH regression head
        ph_head=dict(
            type='SimpleRegHead',
            in_channels=256,
            out_dim=1,  # pH value prediction
            reg_type='ph',
            loss_weight=0.1),
        # VBN regression head
        vbn_head=dict(
            type='SimpleRegHead',
            in_channels=256,
            out_dim=1,  # VBN value prediction
            reg_type='vbn',
            loss_weight=0.1)
    )
)

# Override dataset settings
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
            dict(type='PackDetInputs', 
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction', 
                           'ph_value', 'vbn_value', 'clf_score'))
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
            dict(type='PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'ph_value', 'vbn_value', 'clf_score'))
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
            dict(type='PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'ph_value', 'vbn_value', 'clf_score'))
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

# Override training settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=5)

# Override optimizer settings - learning rate 줄이기
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),  # gradient clipping 추가
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        bias_lr_mult=2.0,
        custom_keys=dict(
            backbone=dict(decay_mult=0.0, lr_mult=0.0),
            neck=dict(decay_mult=0.0, lr_mult=0.0),
            # 새로운 heads의 learning rate를 더 낮게 설정
            bbox_head=dict(decay_mult=1.0, lr_mult=0.1)),
        norm_decay_mult=0.0),
    type='OptimWrapper')

# Learning rate scheduler
param_scheduler = [
    dict(  
        type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=100,
        by_epoch=True,
        eta_min=0)
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