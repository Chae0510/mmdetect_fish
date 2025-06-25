#_base_ = '/workspace/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
_base_ = '/workspace/mmdetect_fish/mmdetection/rtmdet_tiny_8xb32-300e_coco.py'

dataset_type = 'CocoDataset'
classes = ('whole_body', 'eye', 'gill')
data_root = '/workspace/fish_data/'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(
        in_channels=96, feat_channels=96,
        exp_on_reg=False,
        num_classes=3)


)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    # dict(
    #     type='RandomResize',
    #     scale=(1280, 1280),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),

    #dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train_annotation_nocut.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val_annotation_nocut.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

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
    optimizer=dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,  
        by_epoch=True,
        begin=0,
        end=5
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100
    )
]

randomness = dict(
    seed=42,
    deterministic=True,
    diff_rank_seed=False
)