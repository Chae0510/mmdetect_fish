#!/bin/bash

# Set environment variables
export PYTHONPATH=/workspace/mmdetect_fish/mmdetection_head:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Create work directory
WORK_DIR=/workspace/mmdetect_fish/mmdetection_head/work_dirs/wholebody_simple_multi_task_vfnet
LOG_DIR=/workspace/mmdetect_fish/mmdetection_head/log
mkdir -p $WORK_DIR

# Training command for Simple Multi-Task VFNet with Wholebody data
nohup python /workspace/mmdetect_fish/mmdetection_head/tools/train.py \
    /workspace/mmdetect_fish/mmdetection_head/configs/simple_multi_task_vfnet_wholebody.py \
    --work-dir $WORK_DIR > $LOG_DIR/train.log

# Optional: Resume from checkpoint
# --resume $WORK_DIR/latest.pth

# Optional: Load pretrained weights (already set in config)
# --load-from /workspace/mmdetect_fish/mmdetection/work_dirs/bcy_vfnet_r50_loss_cls_false/epoch_100.pth

# Optional: Run with multiple GPUs
# python /workspace/mmdetect_fish/mmdetection_head/tools/train.py \
#     /workspace/mmdetect_fish/mmdetection_head/configs/simple_multi_task_vfnet_wholebody.py \
#     --work-dir $WORK_DIR \
#     --amp \
#     --launcher pytorch

# Optional: Enable auto learning rate scaling
# --auto-scale-lr

# Optional: Test the model after training
echo "Training completed. Running test..."
nohup python /workspace/mmdetect_fish/mmdetection_head/tools/test.py \
    /workspace/mmdetect_fish/mmdetection_head/configs/simple_multi_task_vfnet_wholebody.py \
    $WORK_DIR/latest.pth \
    --show-dir $WORK_DIR/test_results > $LOG_DIR/test.log
