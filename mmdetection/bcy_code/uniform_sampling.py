import json
import random
from collections import defaultdict

input_path = "/workspace/fish_data/all.json"
output_path = "/workspace/fish_data/vbn_ph_uniform_4000.json"
target_total = 4000

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])

grouped = defaultdict(list)
for item in images:
    clf_score = item.get("clf_score")
    if not clf_score or len(clf_score) < 1:
        continue
    try:
        freshness = int(clf_score[-1])
        grouped[freshness].append(item)
    except:
        continue

target_scores = sorted(grouped.keys())
target_per_class = target_total // len(target_scores)

print("샘플링 대상 점수:", target_scores)
print(f"점수당 {target_per_class}개씩 균등 샘플링 (총 {target_total}개 예정)")

sampled = []
for score in target_scores:
    group = grouped[score]
    if len(group) < target_per_class:
        print(f"  - 점수 {score}: {len(group)}개밖에 없음 → 전부 사용")
        sampled.extend(group)
    else:
        sampled.extend(random.sample(group, target_per_class))
        print(f"  - 점수 {score}: {target_per_class}개 샘플링")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"images": sampled}, f, ensure_ascii=False, indent=2)

print(f"\n총 {len(sampled)}개 균형 샘플 저장 완료: {output_path}")
