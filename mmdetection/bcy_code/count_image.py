from pathlib import Path

image_dir = "/workspace/ori_images"
image_files = list(Path(image_dir).glob("*.jpg"))

object_ids = set()
skipped = 0

for img_path in image_files:
    name = img_path.stem  
    parts = name.split("_")

    if len(parts) == 4:
        object_id = parts[1] + "_" + parts[2] 
        object_ids.add(object_id)
    else:
        skipped += 1
        print(f"잘못된 파일: {img_path.name} → {parts}")

print(f"전체 이미지: {len(image_files)}")
print(f"날짜+시간 기준으로 분리 -> 객체로 count: {len(object_ids)}")
print(f"객체당 평균 이미지 개수: {len(image_files) / len(object_ids):.2f}")
