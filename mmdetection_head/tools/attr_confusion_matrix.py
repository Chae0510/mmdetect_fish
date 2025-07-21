#!/usr/bin/env python
"""Compute confusion matrices for the 8×5-class attribute heads.

Usage
-----
python attr_confusion_matrix.py <results.pkl> [--csv <out.csv>] [--png <out.png>] [--per-head-png <dir>]

Arguments
---------
<results.pkl>   Path to the pickled list returned by ``tools/test.py --out ...``.
--csv           (optional) CSV file to save the aggregated 5×5 matrix.
--png           (optional) Save aggregated matrix as a heat-map PNG (requires seaborn).
--per-head-png  (optional) Directory to save a heat-map PNG for every head.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

# Optional deps for nice heat-maps
try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

N_HEADS = 8
N_CLS = 5  # 5-class scores (values 1‥5 in GT → 0‥4 internally)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribute-head confusion-matrix")
    p.add_argument("pkl", help="results pickle produced by MMDet test.py --out …")
    p.add_argument("--csv", help="save aggregated confusion matrix to CSV file")
    p.add_argument("--png", help="save aggregated heat-map to a PNG (requires seaborn)")
    p.add_argument("--per-head-png", help="directory to save per-head heat-maps (requires seaborn)")
    return p.parse_args()

def load_results(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def accumulate_confusion(samples: List):
    cms = np.zeros((N_HEADS, N_CLS, N_CLS), dtype=int)
    skipped = 0
    for s in samples:
        # Unified access for both DetDataSample objects and plain dicts
        if hasattr(s, "metainfo"):
            meta = s.metainfo
            preds = s.pred_instances
        else:
            # Plain dict: meta fields may be at top level
            if "metainfo" in s:
                meta = s["metainfo"]  # type: ignore[index]
            else:
                meta = s  # type: ignore[assignment]
            preds = s.get("pred_instances", None)  # type: ignore[index]

        gt_raw = meta.get("clf_score", None)
        if preds is None:
            logits = None
        elif isinstance(preds, dict):
            logits = preds.get("clf_logits", None)
        else:  # InstanceData or similar
            logits = getattr(preds, "clf_logits", None)

        if gt_raw is None or logits is None or logits.numel() == 0:
            skipped += 1
            continue

        gt = (np.asarray(gt_raw, dtype=int) - 1).clip(0, N_CLS - 1)  # 0-based, safe
        pred = logits[0].argmax(dim=-1).cpu().numpy()

        if gt.shape[0] != N_HEADS or pred.shape[0] != N_HEADS:
            skipped += 1
            continue

        for h in range(N_HEADS):
            cms[h, gt[h], pred[h]] += 1
    return cms, skipped

def save_csv(cm: np.ndarray, path: str):
    np.savetxt(path, cm, fmt="%d", delimiter=",")
    print(f"[✓] CSV saved → {path}")

def plot_heatmap(cm: np.ndarray, title: str, out_path: str):
    if not _HAS_SNS:
        print("seaborn not installed; skipping heat-map for", title)
        return
    sns.set(font_scale=0.9)
    labels = [str(i) for i in range(1, N_CLS + 1)]
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[✓] PNG saved → {out_path}")

def main():
    args = parse_args()
    samples = load_results(args.pkl)
    cms, skipped = accumulate_confusion(samples)

    print(f"Loaded {len(samples)} samples (skipped {skipped} lacking attribute data)")

    # aggregated matrix across heads
    cm_all = cms.sum(axis=0)
    print("\nAggregated 5×5 confusion matrix (all heads):\n", cm_all)

    # optional outputs
    if args.csv:
        save_csv(cm_all, args.csv)

    if args.png:
        plot_heatmap(cm_all, "All heads", args.png)

    if args.per_head_png:
        Path(args.per_head_png).mkdir(parents=True, exist_ok=True)
        for h in range(N_HEADS):
            out = os.path.join(args.per_head_png, f"head{h}.png")
            plot_heatmap(cms[h], f"Head {h}", out)

if __name__ == "__main__":
    main() 