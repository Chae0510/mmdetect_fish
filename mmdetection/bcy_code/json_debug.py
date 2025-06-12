import json
from collections import defaultdict

def check_annotation_count_per_image(coco_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # 이미지 ID → 파일명 매핑
    id_to_name = {img['id']: img['file_name'] for img in coco['images']}

    # 이미지 ID별 annotation 개수 저장
    image_ann_count = defaultdict(int)
    for ann in coco['annotations']:
        image_ann_count[ann['image_id']] += 1

    for image_id, count in image_ann_count.items():
        name = id_to_name[image_id]
        print(f"{name}: {count}개 bbox")

    # bbox가 3개가 아닌 이미지 출력
    print("\n[이상치] bbox가 3개가 아닌 이미지 목록:")
    for image_id, count in image_ann_count.items():
        if count != 3:
            print(f"  - {id_to_name[image_id]}: {count}개")

if __name__ == "__main__":
    check_annotation_count_per_image(
        coco_json_path="/workspace/mmdetection/bcy_code/converted_bbox_only.json"
    )
 