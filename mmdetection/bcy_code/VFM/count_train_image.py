import json
from collections import defaultdict

ann_path = '/workspace/fish_data/train_annotation_nocut.json'

with open(ann_path, 'r') as f:
    data = json.load(f)

# 1. annotation에 등장한 image_id 수 세기
used_image_ids = set(ann['image_id'] for ann in data['annotations'])

# 2. 전체 image 개수와 실제 사용된 image 개수 출력
total_images = len(data['images'])
used_images = len(used_image_ids)

print(f'총 등록된 이미지 수: {total_images}')
print(f'실제로 학습에 사용된 이미지 수 (bbox 있는 이미지): {used_images}')
print(f'제외된 이미지 수: {total_images - used_images}')
