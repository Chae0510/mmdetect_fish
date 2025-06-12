import os
import json
import shutil

def copy_images_from_coco(coco_json_path, image_root_dir, save_dir):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_names = [img["file_name"] for img in coco["images"]]
    os.makedirs(save_dir, exist_ok=True)

    copied, not_found = 0, 0

    for name in image_names:
        found = False
        for root, _, files in os.walk(image_root_dir):
            if name in files:
                src = os.path.join(root, name)
                dst = os.path.join(save_dir, name)
                shutil.copyfile(src, dst)
                copied += 1
                found = True
                break
        if not found:
            print(f"못 찾은 이미지: {name}")
            not_found += 1

    print(f"이미지 복사 완료: {copied}개")
    if not_found > 0:
        print(f"못 찾은 이미지: {not_found}개")

if __name__ == "__main__":
    copy_images_from_coco(
        coco_json_path="/workspace/mmdetection/bcy_code/annotation.json",
        save_dir="/workspace/mmdetection/data/bcy_fish/train/images/",  
        image_root_dir="/workspace/images/fish_data/" 
    )
