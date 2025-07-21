from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.logging import MMLogger

# Register to MMEngine HOOKS so it can be built from cfg

@HOOKS.register_module()
class AttrLossUnfreezeHook(Hook):
    """At a given epoch, restore attribute loss weights that were zeroed
    for detection-only warm-up.

    Args:
        trigger_epoch (int): epoch index (1-based) at which to set the new
            loss weights.
        ph_w, vbn_w, clf_w, clf_last_w (float): values to assign.
    """

    priority = 'VERY_LOW'

    def __init__(self, trigger_epoch: int, *,
                 ph_w: float, vbn_w: float,
                 clf_w: float, clf_last_w: float) -> None:
        self.trigger_epoch = trigger_epoch
        self.ph_w = ph_w
        self.vbn_w = vbn_w
        self.clf_w = clf_w
        self.clf_last_w = clf_last_w
        self._done = False

    def before_train_epoch(self, runner: Runner) -> None:  # type: ignore[override]
        if self._done:
            return
        cur_epoch = runner.epoch
        if cur_epoch + 1 >= self.trigger_epoch:  # epoch is 0-based internally
            head = getattr(runner.model, 'bbox_head', None)
            if head is not None:
                head.ph_w = self.ph_w  # type: ignore[attr-defined]
                head.vbn_w = self.vbn_w
                head.clf_w = self.clf_w
                head.clf_last_w = self.clf_last_w
                logger: MMLogger = runner.logger
                logger.info(f'AttrLossUnfreezeHook: restored attr loss weights '
                            f'ph={self.ph_w}, vbn={self.vbn_w}, clf={self.clf_w}, '
                            f'clf_last={self.clf_last_w} at epoch {cur_epoch + 1}')
                self._done = True 