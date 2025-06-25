_base_ = '/workspace/mmdetect_fish/mmdetection/configs/vfnet/vfnet_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
classes = ('whole_body', 'eye', 'gill')
data_root = '/workspace/fish_data/'

# 학습된 VFNet checkpoint 사용
load_from = '/workspace/mmdetect_fish/mmdetection/work_dirs/bcy_vfnet_r50_loss_cls_false/epoch_100.pth'

model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,  # 모든 backbone stages를 고정
        norm_cfg=dict(type='BN', requires_grad=False),  # backbone의 BN도 고정
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        #frozen=True  # FPN도 고정
    ),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=False,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)
    )
)

metainfo = dict(
    classes=classes,
    palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train_annotation_nocut.json',
        data_prefix=dict(img='train/'),
        
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val_annotation_nocut.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test_annotation_nocut.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_annotation_nocut.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test_annotation_nocut.json',
    metric='bbox',
    classwise=True
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        save_last=True
    )
)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005),
    paramwise_cfg=dict(
        norm_decay_mult=0., 
        bias_decay_mult=0.,
        # backbone과 neck 파라미터 제외
        custom_keys={
            'backbone': dict(lr_mult=0.0, decay_mult=0.0),
            'neck': dict(lr_mult=0.0, decay_mult=0.0),
        }
    )
)

param_scheduler = [
    dict(  
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5
    ),
    dict(  # cosine annealing
        type='CosineAnnealingLR',
        eta_min=1e-5,     
        T_max=95,         
        by_epoch=True,
        begin=5,
        end=100
    )
]