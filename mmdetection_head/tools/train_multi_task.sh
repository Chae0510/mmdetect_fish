#!/bin/bash

# Set environment variables
export PYTHONPATH=/workspace/mmdetect_fish/mmdetection_head:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Create work directory
WORK_DIR=/workspace/mmdetect_fish/mmdetection_head/work_dirs/multi_task_rcnn
mkdir -p $WORK_DIR

# Training command
python /workspace/mmdetect_fish/mmdetection_head/tools/train.py \
    /workspace/mmdetect_fish/mmdetection_head/configs/faster_rcnn/multi_task_rcnn_r50_fpn.py \
    --work-dir $WORK_DIR \
    --amp \
    --seed 42 \
    --gpu-id 0 \
    --deterministic

# Optional: Resume from checkpoint
# --resume $WORK_DIR/latest.pth

# Optional: Load pretrained weights
# --load-from /path/to/pretrained/weights.pth

# Optional: Run with multiple GPUs
# python /workspace/mmdetect_fish/mmdetection_head/tools/train.py \
#     /workspace/mmdetect_fish/mmdetection_head/configs/faster_rcnn/multi_task_rcnn_r50_fpn.py \
#     --work-dir $WORK_DIR \
#     --amp \
#     --seed 42 \
#     --gpus 2 \
#     --deterministic 