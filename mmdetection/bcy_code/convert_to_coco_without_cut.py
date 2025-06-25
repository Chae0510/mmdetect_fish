import json
import os
from pathlib import Path
from PIL import Image

def convert_subset_to_coco_with_size(
    data, image_dir, output_path, file_list, part_categories
):
    coco = {
        "images": [],
        "annotations": [],
        "categories": part_categories
    }

    annotation_id = 1
    image_id = 1
    skipped = 0
    added = 0

    for filename in file_list:
        meta = next((v for k, v in data.items() if Path(k).name == filename), None)
        if not meta:
            print(f"[스킵] {filename}: 메타데이터 없음")
            skipped += 1
            continue

        if meta.get("image_type") == "절단면":
            skipped += 1
            continue  # 절단면 제외

        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
        except Exception as e:
            print(f"[스킵] {filename}: 이미지 열기 실패: {e}")
            skipped += 1
            continue

        points = meta.get("points", [])
        if not isinstance(points, list) or len(points) != 3:
            print(f"[스킵] {filename}: bbox 3개 아님 (len={len(points)})")
            skipped += 1
            continue

        valid_boxes = []
        for i, box in enumerate(points):
            if not isinstance(box, list) or len(box) < 6:
                continue
            try:
                x1, y1, _, x2, y2, _ = box
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                valid_boxes.append({
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "category_id": i + 1
                })
            except Exception as e:
                continue

        if len(valid_boxes) != 3:
            print(f"[스킵] {filename}: bbox 유효한 게 3개 아님")
            skipped += 1
            continue

        coco["images"].append({
            "file_name": filename,
            "id": image_id,
            "width": width,
            "height": height
        })

        for box in valid_boxes:
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "bbox": box["bbox"],
                "area": box["area"],
                "iscrowd": 0,
                "category_id": box["category_id"]
            })
            annotation_id += 1

        image_id += 1
        added += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {output_path}")
    print(f"이미지 수: {added}")
    print(f"평균 bbox 수: {len(coco['annotations']) / added if added else 0:.2f}")
    print(f"스킵된 항목 수: {skipped}")


if __name__ == "__main__":
    with open('/workspace/fish_data/ori_json.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories_nocut = [
        {"id": 1, "name": "whole_body"},
        {"id": 2, "name": "eye"},
        {"id": 3, "name": "gill"}
    ]

    split_base = '/workspace/mmdetect_fish/mmdetection/bcy_code/after_split'

    for split in ['train', 'val', 'test']:
        with open(os.path.join(split_base, f"{split}_files.txt"), "r") as f:
            file_list = [line.strip() for line in f]

        convert_subset_to_coco_with_size(
            data=data,
            image_dir=f'/workspace/fish_data/{split}',
            output_path=f'/workspace/fish_data/{split}_annotation_nocut.json',
            file_list=file_list,
            part_categories=categories_nocut
        )
