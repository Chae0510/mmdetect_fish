#!/bin/bash

# Simple test launcher (matching minimal train script style)
# ----------------------------------------------------------
# Usage: ./test_bbox_level_task_vfnet.sh <checkpoint_path>
# ----------------------------------------------------------

# Environment
export PYTHONPATH=/workspace/mmdetect_fish/mmdetection_head:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1  # first GPU

CFG=/workspace/mmdetect_fish/mmdetection_head/configs/bbox_level_multi_task_vfnet_wholebody_crop.py
LOG_DIR=/workspace/mmdetect_fish/mmdetection_head/log
mkdir -p $LOG_DIR

# -------- user-editable checkpoint path --------------------
# If you prefer hard-coding the ckpt, just edit the line below.
# Leave it empty ("") to supply path as CLI arg or auto-lookup.
CKPT="/workspace/mmdetect_fish/mmdetection_head/work_dirs/bbox_level_multi_task_vfnet_wholebody_crop/last_checkpoint.pth"

# Run evaluation
echo "Starting bbox-level TEST ..."
nohup python /workspace/mmdetect_fish/mmdetection_head/tools/test.py \
    $CFG $CKPT --launcher none --out $LOG_DIR/test_results.pkl \
    > $LOG_DIR/bbox_level_test.log

echo "Logs saved to $LOG_DIR/bbox_level_test.log"
echo "Results pickle: $LOG_DIR/test_results.pkl" 