auto_scale_lr = dict(base_batch_size=4, enable=True)
custom_hooks = [
    dict(interval=1, score_thr=0.3, type='GTRescaleVisHook'),
    dict(type='TrainEvalHook'),
    dict(
        clf_last_w=6.0,
        clf_w=1.0,
        ph_w=0.5,
        trigger_epoch=6,
        type='AttrLossUnfreezeHook',
        vbn_w=0.2),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmdet.engine.hooks.gt_rescale_vis_hook',
        'mmdet.engine.hooks.train_eval_hook',
        'mmdet.engine.hooks.attr_loss_unfreeze_hook',
        'mmdet.evaluation.metrics.attr_metric',
    ])
data_root = '/workspace/20250611_bcy/'
dataset_type = 'CustomCocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        clf_last_loss_weight=0.0,
        clf_loss_weight=0.0,
        feat_channels=256,
        in_channels=256,
        last_head_class_weights=[
            1.0,
            2.0,
            0.25,
            3.0,
            4.0,
        ],
        loss_bbox=dict(loss_weight=1.5, type='GIoULoss'),
        loss_bbox_refine=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0,
            type='VarifocalLoss',
            use_sigmoid=True),
        num_classes=3,
        ph_loss_weight=0.0,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='BboxLevelVFNetHead',
        use_atss=True,
        use_vfl=True,
        vbn_loss_weight=0.0),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=9, type='ATSSAssigner'),
        debug=False,
        pos_weight=-1),
    type='VFNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.008, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(begin=0, by_epoch=True, end=15, eta_min=0, type='CosineAnnealingLR'),
]
resume = False
test_ann = 'dataset/mackerel/splits/test.json'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='dataset/mackerel/splits/test.json',
        data_prefix=dict(img='ori_images/'),
        data_root='/workspace/20250611_bcy/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(ph_div=14.0, type='NormalizeAttr', vbn_div=50.0),
            dict(keep_ratio=True, scale=(
                800,
                600,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'ph_value',
                    'vbn_value',
                    'clf_score',
                    'group_id',
                    'img_type',
                ),
                type='PackDetInputs'),
        ],
        type='CustomCocoDataset'),
    num_workers=8,
    persistent_workers=True)
test_evaluator = [
    dict(
        ann_file='/workspace/20250611_bcy/dataset/mackerel/splits/test.json',
        metric='bbox',
        type='CocoMetric'),
    dict(prefix='test', type='AttrMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(ph_div=14.0, type='NormalizeAttr', vbn_div=50.0),
    dict(keep_ratio=True, scale=(
        800,
        600,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'ph_value',
            'vbn_value',
            'clf_score',
            'group_id',
            'img_type',
        ),
        type='PackDetInputs'),
]
train_ann = 'dataset/mackerel/splits/train.json'
train_cfg = dict(max_epochs=15, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='dataset/mackerel/splits/train.json',
        data_prefix=dict(img='ori_images/'),
        data_root='/workspace/20250611_bcy/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(ph_div=14.0, type='NormalizeAttr', vbn_div=50.0),
            dict(keep_ratio=True, scale=(
                800,
                600,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'ph_value',
                    'vbn_value',
                    'clf_score',
                    'group_id',
                    'img_type',
                ),
                type='PackDetInputs'),
        ],
        type='CustomCocoDataset'),
    num_workers=8,
    persistent_workers=True)
train_evaluator = [
    dict(prefix='train', type='AttrMetric'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(ph_div=14.0, type='NormalizeAttr', vbn_div=50.0),
    dict(keep_ratio=True, scale=(
        800,
        600,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'ph_value',
            'vbn_value',
            'clf_score',
            'group_id',
            'img_type',
        ),
        type='PackDetInputs'),
]
val_ann = 'dataset/mackerel/splits/val.json'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='dataset/mackerel/splits/val.json',
        data_prefix=dict(img='ori_images/'),
        data_root='/workspace/20250611_bcy/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(ph_div=14.0, type='NormalizeAttr', vbn_div=50.0),
            dict(keep_ratio=True, scale=(
                800,
                600,
            ), type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'ph_value',
                    'vbn_value',
                    'clf_score',
                    'group_id',
                    'img_type',
                ),
                type='PackDetInputs'),
        ],
        type='CustomCocoDataset'),
    num_workers=8,
    persistent_workers=True)
val_evaluator = [
    dict(
        ann_file='/workspace/20250611_bcy/dataset/mackerel/splits/val.json',
        metric='bbox',
        type='CocoMetric'),
    dict(prefix='val', type='AttrMetric'),
]
vis_backends = [
    dict(save_dir='${work_dir}/vis', type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(save_dir='${work_dir}/vis', type='LocalVisBackend'),
    ])
work_dir = '/workspace/20250611_bcy/mmdetect_fish/mmdetection_head/work_dirs/vfnet_multi_task_mackerel_nocrop'
