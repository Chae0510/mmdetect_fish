# Use standard VFNet R50 FPN base model, default schedule & runtime only
_base_ = [
    './_base_/models/vfnet_r50_fpn.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------

dataset_type = 'CustomCocoDataset'
# Root directory that contains both `ori_images/` and `dataset/mackerel/`.
data_root = '/workspace/20250611_bcy/'

# Annotation jsons will be referenced relative to data_root.
train_ann = 'dataset/mackerel/splits/train.json'
val_ann   = 'dataset/mackerel/splits/val.json'
test_ann  = 'dataset/mackerel/splits/test.json'

# -----------------------------------------------------------------------------
# Pipelines (WholeBodyCrop removed)
# -----------------------------------------------------------------------------

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NormalizeAttr', ph_div=14.0, vbn_div=50.0),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',
                    'flip','flip_direction','ph_value','vbn_value','clf_score',
                    'group_id','img_type'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NormalizeAttr', ph_div=14.0, vbn_div=50.0),
    dict(type='Resize', scale=(800, 600), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor',
                    'flip','flip_direction','ph_value','vbn_value','clf_score',
                    'group_id','img_type'))
]

# -----------------------------------------------------------------------------
# Dataloaders (override ann_file / pipeline)
# -----------------------------------------------------------------------------

train_dataloader = dict(
    batch_size=4,          # per-GPU batch size
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img='ori_images/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img='ori_images/'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_ann,
        data_prefix=dict(img='ori_images/'),
        pipeline=test_pipeline))

# -----------------------------------------------------------------------------
# Evaluators
# -----------------------------------------------------------------------------
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + val_ann, metric='bbox'),
    dict(type='AttrMetric', prefix='val')
]

# Optional: keep consistency for test set as well
test_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + test_ann, metric='bbox'),
    dict(type='AttrMetric', prefix='test')
]

# Evaluate on training set each epoch to log classification confusion matrix
train_evaluator = [
    dict(type='AttrMetric', prefix='train')
]

# -----------------------------------------------------------------------------
# Model tweaks: 3 classes (id 2,3,4)
# -----------------------------------------------------------------------------
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='BboxLevelVFNetHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        use_atss=True,
        use_vfl=True,
        # Start with zero weights; restored by AttrLossUnfreezeHook at epoch 6
        ph_loss_weight=0.0,
        vbn_loss_weight=0.0,
        clf_loss_weight=0.0,
        clf_last_loss_weight=0.0,
        # Re-weight last classification head (classes 1 … 5)
        #   1: baseline 1.0 (majority)
        #   2: slightly rarer → 2.0
        #   3: very frequent  → 0.25 (down-weight)
        #   4: rare          → 3.0
        #   5: rarest        → 4.0
        last_head_class_weights=[1.0, 2.0, 0.25, 3.0, 4.0],
        loss_cls=dict(type='VarifocalLoss', use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
    ))

# -----------------------------------------------------------------------------
# Checkpoint & work dir
# -----------------------------------------------------------------------------
load_from = None
resume = False
work_dir = './work_dirs/vfnet_mackerel_nocrop'

# -----------------------------------------------------------------------------
# Optimizer adjustments for single-GPU, small-batch training
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,  # replace settings from the base schedule
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001),  # lr scaled 4x
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Automatically scale LR if batch size changes further
auto_scale_lr = dict(enable=True, base_batch_size=4)  # 1 GPU × 4 imgs

# -----------------------------------------------------------------------------
# Visualization: save a few validation images with predictions each epoch
# -----------------------------------------------------------------------------
# Save results under <work_dir>/vis/  (LocalVisBackend + DetVisualizationHook)
vis_backends = [dict(type='LocalVisBackend', save_dir='${work_dir}/vis')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Override default_hooks.visualization: disable draw to avoid duplicate
default_hooks = dict(
    visualization=dict(type='DetVisualizationHook', draw=False)
)

# Custom hook to rescale GT boxes for visualization
custom_hooks = [
    dict(type='GTRescaleVisHook', interval=1, score_thr=0.3),
    dict(type='TrainEvalHook'),
    # Restore attribute loss weights after 5 warm-up epochs
    dict(type='AttrLossUnfreezeHook', trigger_epoch=6,
         ph_w=0.5, vbn_w=0.2, clf_w=1.0, clf_last_w=6.0),
]

# -----------------------------------------------------------------------------
# Custom imports (register custom hooks & metrics)
# -----------------------------------------------------------------------------
custom_imports = dict(
    imports=[
        'mmdet.engine.hooks.gt_rescale_vis_hook',
        'mmdet.engine.hooks.train_eval_hook',
        'mmdet.engine.hooks.attr_loss_unfreeze_hook',
        'mmdet.evaluation.metrics.attr_metric',
    ],
    allow_failed_imports=False,
) 