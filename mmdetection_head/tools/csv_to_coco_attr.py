#!/usr/bin/env python
"""csv_to_coco_attr.py
Convert mackerel CSV annotations to COCO JSON with attribute fields.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List


def median_int(vals: List[int]) -> int:
    return int(round(statistics.median(vals)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV→COCO converter (with attrs)")
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument("--img-root", required=True, help="Image root prefix")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--height", type=int, default=0, help="Image height (optional)")
    p.add_argument("--width", type=int, default=0, help="Image width (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    img_root = Path(args.img_root)
    out_path = Path(args.out_json)

    images: List[Dict[str, Any]] = []
    anns: List[Dict[str, Any]] = []
    img_id_map: Dict[str, int] = {}
    ann_id = 1

    # Buffer rows per group for timestamp sorting
    buf: Dict[str, List[Dict[str, Any]]] = {}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            buf.setdefault(row.get("group_id", ""), []).append(row)

    ts_pat = re.compile(r"_(\d{8})_(\d{6})_")

    for gid, rows in buf.items():
        # sort by timestamp extracted from filename
        def ts_val(r):
            m = ts_pat.search(r["filename"])
            if m:
                return int(m.group(1) + m.group(2))
            return 0

        rows.sort(key=ts_val)

        for idx, row in enumerate(rows):
            img_type = "whole" if idx == 0 else "cut"

            # validate annotations list
            try:
                anno_list = ast.literal_eval(row["annotations"])
            except Exception:
                continue  # skip if cannot parse
            if not (isinstance(anno_list, list) and len(anno_list) == 8 and all(isinstance(h, (list, tuple)) and len(h) >= 1 for h in anno_list)):
                # skip samples with missing annotator scores
                continue

            fname = row["filename"].strip()
            if fname not in img_id_map:
                iid = len(img_id_map) + 1
                img_id_map[fname] = iid

                # take median per head (length >=1)
                clf_medians = [median_int([int(x) for x in head]) for head in anno_list]

                images.append({
                    "id": iid,
                    "file_name": str(img_root / fname),
                    "height": args.height,
                    "width": args.width,
                    "ph_value": float(row["ph"]),
                    "vbn_value": float(row["VBN"]),
                    "clf_score": clf_medians,
                    "group_id": gid,
                    "img_type": img_type
                })

            iid = img_id_map[fname]

            # parse bounding boxes
            try:
                bplist = ast.literal_eval(row["bbox_points"])
            except Exception:
                bplist = []
            if not bplist:
                continue

            for bp in bplist:
                if len(bp) != 6:
                    continue
                xmin, ymin, _clsA, xmax, ymax, cls_id = bp
                w, h = xmax - xmin, ymax - ymin
                anns.append({
                    "id": ann_id,
                    "image_id": iid,
                    "category_id": int(cls_id),
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

    cat_ids = sorted({a["category_id"] for a in anns})
    categories = [{"id": cid, "name": f"class_{cid}"} for cid in cat_ids]

    coco = {"images": images, "annotations": anns, "categories": categories}
    out_path.write_text(json.dumps(coco))
    print(f"[✓] Saved {len(images)} images, {len(anns)} annotations → {out_path}")


if __name__ == "__main__":
    main() 