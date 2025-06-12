import json
import matplotlib.pyplot as plt

log_path = '/workspace/mmdetection/work_dirs/bcy_rtmdet_tiny/20250605_043921/vis_data/20250605_043921.json'

iterations = []
total_loss = []
cls_loss = []
bbox_loss = []

# JSON Lines 형식으로 읽기
with open(log_path, 'r') as f:
    for line in f:
        try:
            entry = json.loads(line)
            if 'loss' in entry:
                iterations.append(entry['step'])
                total_loss.append(entry['loss'])
                cls_loss.append(entry.get('loss_cls', 0))
                bbox_loss.append(entry.get('loss_bbox', 0))
        except json.JSONDecodeError:
            continue  

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(iterations, total_loss, label='Total Loss')
plt.plot(iterations, cls_loss, label='Classification Loss')
plt.plot(iterations, bbox_loss, label='BBox Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
