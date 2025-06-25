import json
import os
from pathlib import Path
from PIL import Image

def extract_vbn_ph_per_annotator(data, image_dirs, output_path_base):
    output = {"images": []}
    img_id = 1

    annotation_keys = ["형태", "색깔", "눈", "절단면", "선별", "건조 및 유소", "이취", "신선도"]

    file_list = [Path(k).name for k in data.keys()]

    for filename in file_list:
        meta_key = next((k for k in data.keys() if Path(k).name == filename), None)
        if not meta_key:
            continue
        meta = data[meta_key]

        found = False
        for image_dir in image_dirs:
            filepath = os.path.join(image_dir, filename)
            if os.path.exists(filepath):
                found = True
                break

        if not found:
            continue

        try:
            with Image.open(filepath) as img:
                width, height = img.size
        except:
            width, height = 640, 480

        annotations = meta.get("annotations", [])
        if len(annotations) != 8 or not all(len(a) == 3 for a in annotations):
            continue  

        for annotator_idx in range(3):
            clf_score = [annotations[i][annotator_idx] for i in range(8)]

            entry = {
                "file_name": filename,
                "id": img_id,
                "width": width,
                "height": height,
                "VBN": meta.get("VBN", None),
                "PH": meta.get("ph", None),
                "annotator": annotator_idx + 1,
                "clf_score": clf_score
            }

            output["images"].append(entry)

        img_id += 1

    output_path = output_path_base
    count = 1
    while os.path.exists(output_path):
        output_path = output_path_base.replace(".json", f"_v{count}.json")
        count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[✓] Saved to: {output_path}")


if __name__ == "__main__":
    base_path = "/workspace/fish_data"

    with open(os.path.join(base_path, "ori_json.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    image_dirs = [os.path.join(base_path, d) for d in ['train', 'val', 'test']]
    output_path_base = os.path.join(base_path, "all.json")

    extract_vbn_ph_per_annotator(data, image_dirs, output_path_base)
