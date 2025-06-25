import json
import random
from collections import defaultdict

input_path = "/workspace/fish_data/all.json"
output_path = "/workspace/fish_data/vbn_ph_balanced_by_freshness.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])

grouped = defaultdict(list)
for item in images:
    clf_score = item.get("clf_score")
    if not clf_score or not isinstance(clf_score, list) or len(clf_score) < 1:
        continue

    freshness_raw = clf_score[-1]
    try:
        freshness = int(freshness_raw)
        grouped[freshness].append(item)
    except (ValueError, TypeError):
        continue

print("신선도 점수별 개수:")
for score in sorted(grouped.keys()):
    print(f"  - 점수 {score}: {len(grouped[score])}개")

min_class_len = min(len(v) for v in grouped.values())
print(f"\n→ 최소 클래스 수에 맞춰 {min_class_len}개씩 샘플링합니다.")

sampled = []
for score, group in grouped.items():
    if len(group) >= min_class_len:
        sampled.extend(random.sample(group, min_class_len))
    else:
        sampled.extend(group)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"images": sampled}, f, ensure_ascii=False, indent=2)

print(f"\n총 {len(sampled)}개 균형 샘플 저장 완료: {output_path}")
