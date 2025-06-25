import json
import random
from pathlib import Path

def make_subset_coco_json(input_json_path, output_json_path, ratio=0.4, seed=42):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    random.seed(seed)

    total_images = coco['images']
    num_to_select = int(len(total_images) * ratio)
    selected_images = random.sample(total_images, num_to_select)
    selected_image_ids = set(img['id'] for img in selected_images)

    selected_annotations = [
        ann for ann in coco['annotations'] if ann['image_id'] in selected_image_ids
    ]

    subset_coco = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco['categories']
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(subset_coco, f, ensure_ascii=False, indent=2)

    print(f"Subset saved to: {output_json_path} ({len(selected_images)} images)")

if __name__ == "__main__":
    input_json = '/workspace/fish_data/val_annotation_nocut.json'
    output_json = '/workspace/fish_data/val_annotation_nocut_10p.json'
    make_subset_coco_json(input_json, output_json, ratio=0.1)
