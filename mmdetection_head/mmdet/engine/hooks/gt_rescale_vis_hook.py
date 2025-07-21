from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample


@HOOKS.register_module()
class GTRescaleVisHook(Hook):
    """Fix visualization where GT boxes are too small due to resizing.

    The hook rescales ground-truth bounding boxes back to the *original*
    image coordinates using the ``scale_factor`` stored in ``data_sample``.
    It then renders GT and prediction side-by-side via the current
    ``DetLocalVisualizer`` instance. Prediction boxes are already in
    original coordinates so they are left unchanged.
    """

    def __init__(self, interval: int = 50, score_thr: float = 0.3):
        self.interval = interval
        self.score_thr = score_thr

        # Lazy import to avoid heavy deps at start-up
        from mmdet.visualization import DetLocalVisualizer
        self.vis = DetLocalVisualizer.get_current_instance()

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]):  # type: ignore[override]
        total_iter = runner.iter + batch_idx
        if total_iter % self.interval != 0:
            return

        data_sample = copy.deepcopy(outputs[0].cpu())

        # Rescale gt_instances back to original image size
        if 'gt_instances' in data_sample and 'scale_factor' in data_sample.metainfo:
            scale = data_sample.metainfo['scale_factor']  # (w_scale,h_scale,w_scale,h_scale)
            if isinstance(scale, torch.Tensor):
                scale = scale.cpu().numpy()
            w_scale, h_scale = scale[0], scale[1]
            gt_bboxes = data_sample.gt_instances.bboxes  # tensor Nx4
            gt_bboxes[:, 0] /= w_scale
            gt_bboxes[:, 2] /= w_scale
            gt_bboxes[:, 1] /= h_scale
            gt_bboxes[:, 3] /= h_scale
            data_sample.gt_instances.bboxes = gt_bboxes

        # Load original image bytes
        import mmcv
        img_path = data_sample.img_path
        img = mmcv.imread(img_path, channel_order='rgb')

        out_name = f'val_iter_{total_iter}'
        self.vis.add_datasample(
            name=out_name,
            image=img,
            data_sample=data_sample,
            draw_gt=True,
            draw_pred=True,
            pred_score_thr=self.score_thr,
            show=False,
            wait_time=0,
            out_file=None,
            step=total_iter,
        ) 