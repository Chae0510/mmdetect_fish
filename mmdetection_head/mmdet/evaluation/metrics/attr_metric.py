from __future__ import annotations

"""Custom metric for classification attribute.

Computes **accuracy** for the *last* of 8×5-class `clf_score` heads.

Assumes `DetDataSample.pred_instances` contains
    - `clf_logits` (Tensor(8,5))

and the sample metadata contains
    - `'clf_score'` (List[int] length 8, values 1 … 5)
"""

from typing import Any, Sequence, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
# fmt: off
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
# fmt: on
from mmengine.registry import METRICS
from mmengine.structures import BaseDataElement
from mmdet.registry import METRICS


@METRICS.register_module()
class AttrMetric(BaseMetric):
    """Evaluate pH, VBN (regression) and 8-way classification accuracy.

    The model should add the following fields:
        • data_sample.metainfo['ph_value']   (float)
        • data_sample.metainfo['vbn_value']  (float)
        • data_sample.metainfo['clf_score']  (list[int] length 8)
    And prediction fields:
        • data_sample.pred_instances.ph_pred    (Tensor[N])
        • data_sample.pred_instances.vbn_pred   (Tensor[N])
        • data_sample.pred_instances.clf_logits (Tensor[N,8,5])

    Where N is number of detections; we only use the first element since
    we broadcasted scalar predictions per image.
    """

    default_prefix: Optional[str] = 'attr'

    def __init__(self, prefix: str | None = None, collect_device: str = 'cpu') -> None:
        # MMEngine's BaseMetric supports an optional `prefix` argument which
        # will be used when logging/formatting metric names. Accept it here
        # so that users can pass `prefix='train'|'val'|'test'` via config.
        super().__init__(prefix=prefix, collect_device=collect_device)
        # Use global logger instance if available
        self.logger: MMLogger | None = MMLogger.get_current_instance()
        # Confusion matrix for 5 classes (rows: GT, cols: Pred)
        self.confmat = np.zeros((5, 5), dtype=int)

    # data_batch is unused; predictions is Sequence[DetDataSample]
    def process(self, data_batch: Any, data_samples: Sequence[BaseDataElement]) -> None:  # type: ignore[override]
        """Collect per-sample accuracy for the last classification head."""
        for ds in data_samples:
            # ---------------- Extract meta & predictions -----------------
            if hasattr(ds, 'metainfo'):
                meta = ds.metainfo
                pred_inst = ds.pred_instances
            elif isinstance(ds, dict):
                meta = ds
                pred_inst = ds.get('pred_instances', None)
                if pred_inst is None:
                    continue
            else:
                continue

            # Ground-truth target list
            clf_gt: List[int] = meta.get('clf_score', [])
            if len(clf_gt) != 8:
                continue

            # Predictions
            if isinstance(pred_inst, dict):
                logits_tensor = pred_inst.get('clf_logits', None)
            else:
                logits_tensor = getattr(pred_inst, 'clf_logits', None)

            if logits_tensor is None or len(logits_tensor) == 0:
                continue

            # Tensor shape [8,5] (we take first det already broadcasted)
            logits = logits_tensor[0]
            clf_pred = logits.argmax(dim=-1).cpu().tolist()

            # Accuracy for the last head only
            target_last = clf_gt[-1]
            pred_last = clf_pred[-1]
            pred_idx = pred_last
            gt_idx = target_last - 1  # targets are 1..5

            # Update confusion matrix
            if 0 <= gt_idx < 5 and 0 <= pred_idx < 5:
                self.confmat[gt_idx, pred_idx] += 1

            acc_cls = float(pred_idx == gt_idx)

            self.results.append(acc_cls)

    def compute_metrics(self, results: list) -> dict:  # type: ignore[override]
        if len(results) == 0:
            return {'clf_acc': np.nan}

        clf_acc = np.mean(results)

        # Save confusion matrix once per evaluation phase (executed on main process)
        self._save_confusion_matrix()

        # Reset confusion matrix for next evaluation phase
        self.confmat.fill(0)

        return {'clf_acc': round(float(clf_acc), 4)}

    def _save_confusion_matrix(self) -> None:
        """Save confusion matrix as .npy and .png inside work_dir."""
        import os
        import datetime

        # Determine save directory from logger (if present) else CWD
        save_dir: str
        if self.logger is not None and hasattr(self.logger, 'handlers'):
            save_dir = None  # type: ignore[assignment]
            for h in self.logger.handlers:  # type: ignore[attr-defined]
                if hasattr(h, 'baseFilename'):
                    save_dir = os.path.dirname(h.baseFilename)  # type: ignore[assignment]
                    break
            if save_dir is None:
                save_dir = os.getcwd()
        else:
            save_dir = os.getcwd()

        # Ensure directory exists (especially if prefix contains path separators)
        os.makedirs(save_dir, exist_ok=True)

        # Sanitize prefix to avoid creating nested directories in filename
        raw_prefix = self.prefix or ''
        phase = raw_prefix.replace('/', '_').replace('\\', '_') or 'attr'

        timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')
        npy_path = os.path.join(save_dir, f'confmat_{phase}_{timestamp}.npy')
        np.save(npy_path, self.confmat)

        # Try to also save a PNG visualization
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns  # seaborn gives nicer visuals but optional

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(self.confmat, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=[f'P{i}' for i in range(1, 6)],
                        yticklabels=[f'T{i}' for i in range(1, 6)])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Ground Truth')
            ax.set_title(f'Confusion Matrix ({phase})')
            png_path = os.path.join(save_dir, f'confmat_{phase}_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close(fig)
            if self.logger is not None:
                self.logger.info(f'Saved confusion matrix to {npy_path} and {png_path}')
        except Exception as e:  # pylint: disable=broad-except
            # If seaborn or matplotlib not available, skip image save
            if self.logger is not None:
                self.logger.info(f'Could not save confusion matrix image: {e}. NPY saved to {npy_path}') 