import cv2
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.datasets import CocoDataset

# config 불러오기
cfg = Config.fromfile('/workspace/mmdetect_fish/mmdetection/bcy_code/bcy_rtmdet_tiny.py')
init_default_scope(cfg.get('default_scope', 'mmdet'))

# dataset 생성
dataset_cfg = cfg.train_dataloader.dataset
dataset = CocoDataset(
    ann_file=dataset_cfg.ann_file,
    data_root=dataset_cfg.data_root,
    data_prefix=dataset_cfg.data_prefix,
    pipeline=dataset_cfg.pipeline,
    metainfo=dataset_cfg.metainfo,
    test_mode=False
)

# 샘플 하나 가져오기
sample = dataset[0]
img = sample['inputs'].permute(1, 2, 0).numpy().astype('uint8').copy()
bboxes = sample['data_samples'].gt_instances.bboxes.numpy()
labels = sample['data_samples'].gt_instances.labels.numpy()
classes = dataset.metainfo['classes']

# bbox 그리기
for box, label in zip(bboxes, labels):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, classes[label], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)

# matplotlib으로 시각화
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig('/workspace/fish_data/output_bbox_visualization.png')  # 이미지 저장
print("✅ 시각화 이미지 저장됨: output_bbox_visualization.png")