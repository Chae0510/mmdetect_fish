from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS, build_from_cfg
from mmengine.runner import Runner
from mmengine.runner.loops import ValLoop


@HOOKS.register_module()
class TrainEvalHook(Hook):
    """Run evaluation on training set after each epoch to save confusion matrix.

    This hook instantiates a ValLoop with the train dataloader and the
    configuration provided in ``runner.cfg.train_evaluator``.
    It runs *after* the standard validation loop so 결과를 덮어쓰지 않습니다.
    """

    priority = 'VERY_LOW'

    def after_train_epoch(self, runner: Runner) -> None:  # type: ignore[override]
        # Build evaluator from cfg only once
        if not hasattr(self, '_train_val_loop'):
            evaluator_cfg = runner.cfg.get('train_evaluator', None)
            if evaluator_cfg is None:
                return  # nothing to do

            evaluator = runner.build_evaluator(evaluator_cfg)
            self._train_val_loop = ValLoop(runner, runner.train_dataloader, evaluator)  # type: ignore[attr-defined]
        # Run evaluation on current model state
        self._train_val_loop.run() 

        # Save training confusion matrix accumulated during the epoch
        if hasattr(runner.model.bbox_head, 'save_train_confmat'):
            # Pass save_dir=None so the function resolves to the same directory
            # as logger handlers, keeping train/val confusion matrices together.
            runner.model.bbox_head.save_train_confmat(logger=runner.logger) 