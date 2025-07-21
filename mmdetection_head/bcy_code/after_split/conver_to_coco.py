import json
import os
from pathlib import Path
from PIL import Image

def convert_subset_to_coco(data, image_dir, output_path, file_list, part_categories):
    coco = {"images": [], "annotations": [], "categories": part_categories}
    annotation_id = 1
    img_id = 1

    for filename in file_list:
        # key matching 개선
        meta = next((v for k, v in data.items() if Path(k).name == filename), None)
        if not meta:
            continue


        filepath = os.path.join(image_dir, filename)
        try:
            with Image.open(filepath) as img:
                width, height = img.size
        except:
            width, height = 640, 480

        coco["images"].append({
            "file_name": filename,
            "id": img_id,
            "width": width,
            "height": height
        })

        image_type = meta.get('image_type', '')
        points = meta.get('points', [])

        if image_type == '절단면':
            for box in points:
                if len(box) >= 6:
                    x1, y1, _, x2, y2, _ = box
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "category_id": 4
                    })
                    annotation_id += 1
        else:
            for i, box in enumerate(points):
                if len(box) >= 6:
                    x1, y1, _, x2, y2, _ = box
                    coco["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "category_id": i + 1
                    })
                    annotation_id += 1

        img_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    with open('/workspace/fish_data/ori_json.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = [
        {"id": 1, "name": "whole_body"},
        {"id": 2, "name": "eye"},
        {"id": 3, "name": "gill"},
        {"id": 4, "name": "cut_surface"}
    ]

    for split in ['train', 'val','test']:
        with open(f"{split}_files.txt", "r") as f:
            file_list = [line.strip() for line in f]

        convert_subset_to_coco(
            data=data,
            image_dir=f'/workspace/fish_data/{split}',
            output_path=f'/workspace/fish_data/{split}_annotation.json',
            file_list=file_list,
            part_categories=categories
        )
