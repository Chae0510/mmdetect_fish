import os
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv

config_file = '/workspace/mmdetection/bcy_code/bcy_rtmdet_tiny.py'
checkpoint_file = '/workspace/mmdetection/work_dirs/bcy_rtmdet_tiny/epoch_13.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta

input_dir = '/workspace/images/zip/zip_01d2cc/98'
output_dir = '/workspace/images/results_zip_01d2cc'
os.makedirs(output_dir, exist_ok=True)

valid_exts = ('.jpg', '.png')

for fname in os.listdir(input_dir):
    if fname.lower().endswith(valid_exts):
        img_path = os.path.join(input_dir, fname)
        image = mmcv.imread(img_path, channel_order='rgb')
        result = inference_detector(model, image)

        out_path = os.path.join(output_dir, fname)
        visualizer.add_datasample(
            name=fname,
            image=image,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            pred_score_thr=0.3,
            out_file=out_path
        )
        print(f"Inference done for: {fname}")
