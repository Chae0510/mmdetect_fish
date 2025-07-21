import json
import os
from pathlib import Path
from PIL import Image

def convert_filtered_json_to_coco_with_size(json_input_path, image_root_dir, output_path):
    with open(json_input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "whole_body"},
            {"id": 2, "name": "eye"},
            {"id": 3, "name": "gill"}
        ]
    }

    annotation_id = 1
    image_id_counter = 1
    skipped = 0
    added = 0

    for filename, meta in data.items():
        fname = Path(filename).name
        image_type = meta.get("image_type", "").strip()

        ##--- 절단면은 일단 제외
        if image_type != "홀바디":
            skipped += 1
            continue

        points = meta.get("points", [])
        if not isinstance(points, list) or len(points) != 3:
            print(f"{fname}: points 개수 {len(points)} (3개 XXX -> 날림)")
            skipped += 1
            continue

        image_path = None
        for root, _, files in os.walk(image_root_dir):
            if fname in files:
                image_path = os.path.join(root, fname)
                break

        if image_path is None:
            print(f"이미지 파일 없음: {fname}")
            skipped += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"이미지 열기 실패: {fname}, {e}")
            skipped += 1
            continue

        valid_bbox = []
        for idx, box in enumerate(points):
            if not isinstance(box, list) or len(box) != 6:
                continue
            try:
                x1, y1, _, x2, y2, _ = box
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                valid_bbox.append({
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "category_id": idx + 1
                })
            except Exception as e:
                continue

        if len(valid_bbox) == 3:
            coco_data["images"].append({
                "file_name": fname,
                "id": image_id_counter,
                "width": width,
                "height": height
            })

            for box_data in valid_bbox:
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id_counter,
                    "bbox": box_data["bbox"],
                    "area": box_data["area"],
                    "iscrowd": 0,
                    "category_id": box_data["category_id"]
                })
                annotation_id += 1

            image_id_counter += 1
            added += 1
        else:
            print(f"{fname}: bbox 3개 XXX")
            skipped += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"추가된 이미지 수: {added}")
    print(f"image 당 평균 bbox 개수: {len(coco_data['annotations']) / added if added else 0:.2f}")


if __name__ == "__main__":
    convert_filtered_json_to_coco_with_size(
        json_input_path="/workspace/images/image_bbox1.json",
        image_root_dir="/workspace/images/fish_data",  
        output_path="/workspace/mmdetection/bcy_code/converted_with_real_size.json"
    )
