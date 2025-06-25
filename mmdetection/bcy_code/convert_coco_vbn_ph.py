import json
import os
from pathlib import Path
from PIL import Image

def extract_vbn_ph_json(data, image_dirs, output_path_base):
    output = {"images": []}
    img_id = 1

    file_list = [Path(k).name for k in data.keys()]

    for filename in file_list:
        meta = data.get(f"images/{filename}", None)
        if not meta:
            continue

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

        output["images"].append({
            "file_name": filename,
            "id": img_id,
            "width": width,
            "height": height,
            "VBN": meta.get("VBN", None),
            "PH": meta.get("ph", None)
            
        })

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

    output_path_base = os.path.join(base_path, "vbn_ph.json")

    extract_vbn_ph_json(data, image_dirs, output_path_base)
