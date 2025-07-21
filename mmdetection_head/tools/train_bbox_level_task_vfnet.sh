#!/bin/bash
trial=1
exp_name=bbox_level_multi_task_vfnet_wholebody_crop
# Set environment variables
export PYTHONPATH=/workspace/mmdetect_fish/mmdetection_head:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Create work directory
WORK_DIR=/workspace/mmdetect_fish/mmdetection_head/work_dirs/${exp_name}/trial${trial}
LOG_DIR=/workspace/mmdetect_fish/mmdetection_head/log
mkdir -p $WORK_DIR
mkdir -p $LOG_DIR

# Training command for Bbox-Level Multi-Task VFNet with Wholebody crop data
echo "Starting bbox-level training with WholeBodyCrop..."
nohup python /workspace/mmdetect_fish/mmdetection_head/tools/train.py \
    /workspace/mmdetect_fish/mmdetection_head/configs/bbox_level_multi_task_vfnet_wholebody_crop.py \
    --work-dir $WORK_DIR > $LOG_DIR/${exp_name}_trial${trial}.log

