import os
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv
from mmengine.structures import InstanceData

# 모델 초기화
config_file = '/workspace/mmdetect_fish/mmdetection/bcy_code/bcy_rtmdet_tiny.py'
checkpoint_file = '/workspace/mmdetect_fish/mmdetection/work_dirs/bcy_rtmdet_tiny/epoch_13.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta

input_dir = '/workspace/fish_data/test'
output_dir = '/workspace/fish_data/top1_vis'
os.makedirs(output_dir, exist_ok=True)

valid_exts = ('.jpg', '.png')

for fname in os.listdir(input_dir):
    if fname.lower().endswith(valid_exts):
        img_path = os.path.join(input_dir, fname)
        image = mmcv.imread(img_path, channel_order='rgb')
        result = inference_detector(model, image)

        instances = result.pred_instances
        if instances.scores.numel() == 0:
            continue  # 예측 없음

        top1_index = instances.scores.argmax().item()
        top1_instance = InstanceData()
        for k in instances.keys():
            top1_instance.set_field(instances.get(k)[top1_index:top1_index+1], k)
        result.pred_instances = top1_instance

        out_path = os.path.join(output_dir, fname)
        visualizer.add_datasample(
            name=fname,
            image=image,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            pred_score_thr=0.0,
            out_file=out_path
        )
        print(f"Top-1 prediction saved: {fname}")
