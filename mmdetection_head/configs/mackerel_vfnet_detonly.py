# Detection-only warm-up: 5 epochs, attribute losses disabled
_base_ = ['./mackerel_vfnet_nocrop.py']

# -----------------------------------------------------------------------------
# Turn off attribute losses (pH, VBN, 8×classification)
# -----------------------------------------------------------------------------
model = dict(
    bbox_head=dict(
        ph_loss_weight       = 0.0,
        vbn_loss_weight      = 0.0,
        clf_loss_weight      = 0.0,
        clf_last_loss_weight = 0.0,
        last_head_class_weights = None,
    )
)

# -----------------------------------------------------------------------------
# Cut training to 5 epochs; run validation only at the end
# -----------------------------------------------------------------------------
train_cfg = dict(max_epochs=5, val_interval=5)  # val once after epoch 5

# -----------------------------------------------------------------------------
# Remove attribute metrics & confusion-matrix hooks to save time
# -----------------------------------------------------------------------------
# keep validation CocoMetric; AttrMetric isn't informative when losses are 0
val_evaluator = [dict(type='CocoMetric', metric='bbox', ann_file=_base_.val_ann) if False else None]  # placeholder

train_evaluator = []  # no train evaluation during warm-up

# Custom hooks: only visualization of GT boxes (optional)
custom_hooks = [
    dict(type='GTRescaleVisHook', interval=1, score_thr=0.3),
] 