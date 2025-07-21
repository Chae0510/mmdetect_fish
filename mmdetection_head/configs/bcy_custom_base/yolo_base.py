_base_ = [
    '../_base_/models/yolov5s.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='YOLOV5',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(type='YOLOV5PAFPN', deepen_factor=0.33, widen_factor=0.5),
    bbox_head=dict(
        type='YOLOV5Head',
        num_classes=1,  # override 가능
        in_channels=[128,256,512],
        out_channels=[128,256,512],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(10,13),(16,30),(33,23)],
                        [(30,61),(62,45),(59,119)],
                        [(116,90),(156,198),(373,326)]],
            strides=[8,16,32]
        ),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        conf_loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        cls_loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        reg_loss=dict(type='GIoULoss', loss_weight=2.0)
    )
)
