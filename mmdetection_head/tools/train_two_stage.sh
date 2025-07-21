#!/usr/bin/env bash

# Two-stage training helper
# 1) detection-only warm-up (5 epochs) using det-only config
# 2) resume full multi-task training for the remaining epochs
#
# Usage:
#   bash tools/train_two_stage.sh [NUM_TOTAL_EPOCHS]
# Default total epochs = 15 (5 warm-up + 10 attribute fine-tune)

set -e

# ---------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------
DET_CFG="configs/mackerel_vfnet_detonly.py"
FULL_CFG="configs/mackerel_vfnet_nocrop.py"

# total epochs desired (optional arg)
TOTAL_EPOCHS=${1:-15}
WARM_EPOCHS=5               # as defined in det-only config

# ---------------------------------------------------------------------
# Stage 1 – detection only
# ---------------------------------------------------------------------
python tools/train.py "$DET_CFG"

# locate the latest checkpoint (epoch_5.pth)
CKPT=$(ls -1t work_dirs/*detonly*/epoch_${WARM_EPOCHS}.pth | head -n1)
if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found after Stage 1" >&2
  exit 1
fi

echo "[Stage 1] Warm-up checkpoint: $CKPT"

# ---------------------------------------------------------------------
# Stage 2 – resume full training
# Update max_epochs in FULL_CFG on-the-fly via MMEngine override (-o)
# new_max = TOTAL_EPOCHS ; val_interval = 1 (default)
# ---------------------------------------------------------------------
python tools/train.py "$FULL_CFG" \
       --resume-from "$CKPT" \
       -o train_cfg.max_epochs=$TOTAL_EPOCHS 