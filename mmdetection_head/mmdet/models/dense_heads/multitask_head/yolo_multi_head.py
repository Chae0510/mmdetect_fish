from .yolo_head import YOLOV3Head
from .attr_head_mixin import MultiTaskHeadMixin  # 우리가 만든 Mixin

from mmdet.registry import MODELS


@MODELS.register_module()
class YOLOV3MultiTaskHead(YOLOV3Head, MultiTaskHeadMixin):
    """YOLOv3 Head with multi-task regression (ph, vbn) and classification (clf_score)."""

    def __init__(self, *args, in_channels, **kwargs):
        super().__init__(*args, in_channels=in_channels, **kwargs)
        self.init_attr_heads(in_channels[0])  # 가장 첫 번째 scale level의 채널 사용

    def forward(self, x):
        # YOLO의 원래 detection 결과
        yolo_outputs = super().forward(x)  # pred_maps,

        # attribute 예측
        ph_list = []
        vbn_list = []
        clf_list = []

        for feat in x:
            ph, vbn, clf = self.forward_attr(feat)
            ph_list.append(ph)
            vbn_list.append(vbn)
            clf_list.append(clf)

        return yolo_outputs, ph_list, vbn_list, clf_list
