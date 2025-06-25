import json, os
from pathlib import Path
from shutil import move

# 경로 지정
ann_path = '/workspace/fish_data/test_annotation_nocut.json'
test_img_dir = Path('/workspace/fish_data/test')
filtered_dir = Path('/workspace/fish_data/test_filtered')
filtered_dir.mkdir(exist_ok=True)

# 1. annotation에 있는 이미지 목록 추출
with open(ann_path, 'r') as f:
    ann = json.load(f)
valid_files = {img['file_name'] for img in ann['images']}

# 2. test 폴더에서 해당 파일만 복사
for fname in os.listdir(test_img_dir):
    if fname in valid_files:
        move(test_img_dir / fname, filtered_dir / fname)
