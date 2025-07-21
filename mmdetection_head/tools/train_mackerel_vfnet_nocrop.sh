#!/bin/bash

# ----------------------------------------------------------------------------
# Train multi-task VFNet (no WholeBodyCrop) on the Mackerel CSV dataset
# ----------------------------------------------------------------------------

# Adjust this if your workspace path differs
MMDET_DIR="/workspace/20250611_bcy/mmdetect_fish/mmdetection_head"
# Config located directly under configs directory
CONFIG="$MMDET_DIR/configs/mackerel_vfnet_nocrop.py"
WORK_DIR="$MMDET_DIR/work_dirs/vfnet_multi_task_mackerel_nocrop"

# GPU setting. By default, use 1 GPU (id 0). Pass GPU ID(s) via env var.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

# Ensure work_dir exists
mkdir -p "$WORK_DIR"

# Add MMDetection to PYTHONPATH so that custom modules are discoverable
export PYTHONPATH="$MMDET_DIR":$PYTHONPATH

# Training command
python "$MMDET_DIR/tools/train.py" \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    "$@"

# Usage examples:
#   bash train_mackerel_vfnet_nocrop.sh              # train on GPU 0
#   CUDA_VISIBLE_DEVICES="0,1" bash train_mackerel_vfnet_nocrop.sh --amp   # mixed-precision on 2 GPUs
#   bash train_mackerel_vfnet_nocrop.sh --resume-from <ckpt.pth> 