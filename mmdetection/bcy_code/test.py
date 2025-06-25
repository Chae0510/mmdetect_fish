import json
import pandas as pd

json_path = "/workspace/fish_data/vbn_ph_uniform_annotator1.json"

with open(json_path, "r") as f:
    data = json.load(f)

image_ids = [img["id"] for img in data["images"]]
anno_image_ids = [ann["image_id"] for ann in data["annotations"]]

# pandas DataFrame으로 정렬
df_images = pd.DataFrame(sorted(image_ids), columns=["image_id_from_images"])
df_annos = pd.DataFrame(sorted(anno_image_ids), columns=["image_id_from_annotations"])

# 비교 결과
print("[✓] 총 images 수:", len(df_images))
print("[✓] 총 annotations 수:", len(df_annos))

# 양쪽에만 있는 ID 확인
only_in_images = set(image_ids) - set(anno_image_ids)
only_in_annos = set(anno_image_ids) - set(image_ids)

print("[✓] images에만 있는 ID 수:", len(only_in_images))
print("[✓] annotations에만 있는 ID 수:", len(only_in_annos))

# 겹치는 ID 수
common_ids = set(image_ids) & set(anno_image_ids)
print("[✓] images.id 와 annotations.image_id 가 일치하는 수:", len(common_ids))

# 예시 일부 출력
if only_in_images:
    print("→ images에만 있는 예시:", list(only_in_images)[:10])
if only_in_annos:
    print("→ annotations에만 있는 예시:", list(only_in_annos)[:10])
