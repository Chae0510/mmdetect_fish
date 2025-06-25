import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

def analyze_annotation_coverage(json_path, image_dir):
    # 디렉토리의 이미지 파일 이름 집합
    image_files = {f.name for f in Path(image_dir).glob("*.jpg")}

    # COCO JSON 로딩
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    annotated_ids = set(ann["image_id"] for ann in coco["annotations"])
    labeled_files = {id_to_name[i] for i in annotated_ids if i in id_to_name}

    # 교집합 / 차집합 계산
    labeled = image_files & labeled_files
    unlabeled = image_files - labeled

    return {
        "total": len(image_files),
        "labeled": len(labeled),
        "unlabeled": len(unlabeled),
        "unlabeled_list": list(unlabeled)
    }

# 경로 설정
base = "/workspace/fish_data"
splits = {
    "train": {
        "json": os.path.join(base, "train_annotation_nocut.json"),
        "img_dir": os.path.join(base, "train")
    },
    "val": {
        "json": os.path.join(base, "val_annotation_nocut.json"),
        "img_dir": os.path.join(base, "val")
    },
    "test": {
        "json": os.path.join(base, "test_annotation_nocut.json"),
        "img_dir": os.path.join(base, "test")
    }
}

# 통계 수집
rows = []
for split, paths in splits.items():
    stats = analyze_annotation_coverage(paths["json"], paths["img_dir"])
    rows.append({
        "Split": split,
        "Total Images": stats["total"],
        "Labeled Images": stats["labeled"],
        "Unlabeled Images": stats["unlabeled"]
    })

# DataFrame 생성 및 출력
df = pd.DataFrame(rows)
print(df.to_markdown(index=False))
df.to_csv("annotation_summary.csv", index=False)
print("\n✅ 요약 CSV 저장 완료: annotation_summary.csv")

# ✅ 시각화 (bar chart)
labels = df["Split"]
labeled_counts = df["Labeled Images"]
unlabeled_counts = df["Unlabeled Images"]

x = range(len(labels))
bar_width = 0.6

plt.figure(figsize=(8, 6))
plt.bar(x, labeled_counts, label="Labeled", color="green")
plt.bar(x, unlabeled_counts, bottom=labeled_counts, label="Unlabeled", color="red")

plt.xticks(x, labels)
plt.ylabel("Image Count")
plt.title("Labeled vs Unlabeled Image Distribution per Split")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("annotation_bar_chart.png")
print("✅ 시각화 저장 완료: annotation_bar_chart.png")
