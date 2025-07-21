exp_name=det_fish_cosine_annealing
nohup python train.py --lr_scheduler cosine_annealing --work-dir /workspace/mmdetect_fish/mmdetection_head/work_dirs/${exp_name} --auto-scale-lr > /workspace/mmdetect_fish/mmdetection_head/log/${exp_name}.out
