_base_ = [
    '../_base_/models/efficientdet_d0.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='EfficientDet',
    backbone=dict(
        type='EfficientDetBackbone',
        model_name='efficientdet-d0',
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='efficientdet_d0.pth')
    )
)
