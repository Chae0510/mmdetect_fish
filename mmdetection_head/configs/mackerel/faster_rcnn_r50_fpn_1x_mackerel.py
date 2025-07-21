# Base config borrowed from COCO Faster R-CNN (MMDetection 3.x)
# path relative to this file
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# -------------------------------------------------------------------------
# Dataset settings
# -------------------------------------------------------------------------
train_csv = '/workspace/20250611_bcy/dataset/mackerel/first_image_annotation.csv'
img_root = '/workspace/20250611_bcy'  # contains the "images/" folder

metainfo = dict(classes=('mackerel',), palette=[(220, 20, 60)])

dataset_type = 'MackerelCsvDataset'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=train_csv,
        data_prefix=dict(img=img_root),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=train_csv,  # using same CSV for quick sanity-check; substitute if you have val CSV
        data_prefix=dict(img=img_root),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=None, metric='bbox')

test_evaluator = val_evaluator

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------
model = dict(
    roi_head=dict(  # update num_classes for all bbox heads
        bbox_head=dict(num_classes=1)))

# -------------------------------------------------------------------------
# Optimiser / Schedules (optional tweaks)
# -------------------------------------------------------------------------
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(max_epochs=12, val_interval=1)

# -------------------------------------------------------------------------
# Work directory
# -------------------------------------------------------------------------
work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_mackerel' 