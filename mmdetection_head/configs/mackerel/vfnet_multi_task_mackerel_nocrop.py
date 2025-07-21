# VFNet + Multi-task heads (PH, VBN) for Mackerel dataset (CSV) without WholeBodyCrop
# -----------------------------------------------------------------------------
# Base components: VFNet R50 FPN, 1x schedule, default runtime
# -----------------------------------------------------------------------------
_base_ = [
    '../_base_/models/vfnet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# -----------------------------------------------------------------------------
# Dataset (custom CSV) paths
# -----------------------------------------------------------------------------

dataset_type = 'MackerelCsvDataset'
train_csv = '/workspace/20250611_bcy/dataset/mackerel/first_image_annotation.csv'
img_root  = '/workspace/20250611_bcy'  # contains images/ ...

# -----------------------------------------------------------------------------
# Pipelines  – WholeBodyCrop removed, keep NormalizeAttr (uses ph_value / vbn_value)
# -----------------------------------------------------------------------------

_common_meta = dict(
    meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',
               'flip','flip_direction','ph_value','vbn_value','clf_score',
               'group_id','img_type'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NormalizeAttr', ph_div=14.0, vbn_div=50.0),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', **_common_meta)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NormalizeAttr', ph_div=14.0, vbn_div=50.0),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='PackDetInputs', **_common_meta)
]

# -----------------------------------------------------------------------------
# Dataloaders
# -----------------------------------------------------------------------------

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=train_csv,
        data_prefix=dict(img=img_root),
        pipeline=train_pipeline,
        filter_cfg=None
    ))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=train_csv,  # TODO: replace with val CSV if available
        data_prefix=dict(img=img_root),
        pipeline=test_pipeline,
        test_mode=True
    ))

test_dataloader = val_dataloader

# -----------------------------------------------------------------------------
# Model – use multi-task VFNet head (PH/VBN regression)
# -----------------------------------------------------------------------------
model = dict(
    _delete_=True,
    type='BboxLevelVFNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0,1,2,3),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256,512,1024,2048], out_channels=256,
              start_level=1, add_extra_convs='on_output', num_outs=5,
              relu_before_extra_convs=True),
    bbox_head=dict(
        type='BboxLevelVFNetHead',
        num_classes=1,                 # single fish class
        in_channels=256,
        feat_channels=256,
        strides=[8,16,32,64,128],
        use_atss=True,
        use_vfl=True,
        ph_loss_weight=0.5,
        vbn_loss_weight=0.2,
        loss_cls=dict(type='VarifocalLoss', use_sigmoid=True, alpha=0.75, gamma=2.0,
                       iou_weighted=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)
    ),
    train_cfg=dict(assigner=dict(type='ATSSAssigner', topk=9)),
    test_cfg=dict(
        nms_pre=1000, min_bbox_size=0, score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6), max_per_img=100)
)

# -----------------------------------------------------------------------------
# Optimiser / schedule tweaks
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, end=24, by_epoch=True, eta_min=1e-6)
]

train_cfg = dict(max_epochs=24, val_interval=1)

# -----------------------------------------------------------------------------
work_dir = './work_dirs/vfnet_multi_task_mackerel_nocrop' 