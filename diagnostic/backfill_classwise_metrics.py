#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
from collections import defaultdict


def _to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def _clip_xyxy(box, w, h):
    x1 = max(0.0, min(float(box[0]), float(w)))
    y1 = max(0.0, min(float(box[1]), float(h)))
    x2 = max(0.0, min(float(box[2]), float(w)))
    y2 = max(0.0, min(float(box[3]), float(h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _compute_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mean(values):
    return sum(values) / len(values) if values else None


def _median(values):
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _resolve_ann_pred_paths(run_dir):
    metrics_dir = os.path.join(run_dir, "metrics")
    ann_path = os.path.join(metrics_dir, "test_ann.json")
    pred_path = os.path.join(metrics_dir, "test_preds.json")
    if os.path.exists(ann_path) and os.path.exists(pred_path):
        return ann_path, pred_path

    ann_path = os.path.join(run_dir, "test_ann.json")
    pred_path = os.path.join(run_dir, "test_preds.json")
    if os.path.exists(ann_path) and os.path.exists(pred_path):
        return ann_path, pred_path

    return None, None


def _find_run_dirs(runs_root):
    run_dirs = set()
    for root, _, files in os.walk(runs_root):
        if "test_ann.json" in files and "test_preds.json" in files:
            if os.path.basename(root) == "metrics":
                run_dir = os.path.dirname(root)
            else:
                run_dir = root
            run_dirs.add(run_dir)
    return sorted(run_dirs)


def compute_classwise_from_coco(ann_path, pred_path, iou_thresh, score_thresh, topk):
    with open(ann_path, "r", encoding="utf-8") as handle:
        ann = json.load(handle)
    with open(pred_path, "r", encoding="utf-8") as handle:
        preds = json.load(handle)

    images = {img["id"]: (img["width"], img["height"]) for img in ann.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in ann.get("categories", [])}

    ann_by_image = defaultdict(list)
    for a in ann.get("annotations", []):
        img_id = a.get("image_id")
        cat_id = a.get("category_id")
        if img_id not in images or cat_id not in categories:
            continue
        ann_by_image[img_id].append(a)

    pred_by_image = defaultdict(list)
    for p in preds:
        img_id = p.get("image_id")
        cat_id = p.get("category_id")
        score = p.get("score", 0.0)
        if img_id not in images or cat_id not in categories:
            continue
        if score < score_thresh:
            continue
        pred_by_image[img_id].append(p)

    stats = defaultdict(lambda: {"gt": 0, "matched": 0, "ious": []})

    for img_id, (w, h) in images.items():
        gts = ann_by_image.get(img_id, [])
        preds_img = pred_by_image.get(img_id, [])

        pred_entries = []
        for p in preds_img:
            box = _clip_xyxy(_to_xyxy(p["bbox"]), w, h)
            if box is None:
                continue
            pred_entries.append((p["category_id"], float(p["score"]), box))

        if pred_entries and topk is not None:
            pred_entries.sort(key=lambda x: x[1], reverse=True)
            pred_entries = pred_entries[: int(topk)]

        for a in gts:
            cat_id = a["category_id"]
            cls_name = categories[cat_id]
            stats[cls_name]["gt"] += 1

            gt_box = _clip_xyxy(_to_xyxy(a["bbox"]), w, h)
            if gt_box is None:
                continue

            same_class = [pe for pe in pred_entries if pe[0] == cat_id]
            if not same_class:
                continue

            best_iou = 0.0
            for _, _, pb in same_class:
                iou = _compute_iou(gt_box, pb)
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= iou_thresh:
                stats[cls_name]["matched"] += 1
                stats[cls_name]["ious"].append(best_iou)

    summary = {
        "config": {
            "iou_thresh": float(iou_thresh),
            "score_thresh": float(score_thresh),
            "topk": int(topk) if topk is not None else None,
        },
        "per_class": {},
    }

    for cat_id in sorted(categories):
        cls_name = categories[cat_id]
        s = stats[cls_name]
        recall = (s["matched"] / s["gt"]) if s["gt"] else 0.0
        mean_iou = _mean(s["ious"]) if s["ious"] else None
        median_iou = _median(s["ious"]) if s["ious"] else None
        iou_ge_070 = (sum(1 for v in s["ious"] if v >= 0.70) / len(s["ious"])) if s["ious"] else None
        iou_ge_085 = (sum(1 for v in s["ious"] if v >= 0.85) / len(s["ious"])) if s["ious"] else None

        summary["per_class"][cls_name] = {
            "gt": int(s["gt"]),
            "matched": int(s["matched"]),
            "recall": float(recall),
            "mean_iou": float(mean_iou) if mean_iou is not None else None,
            "median_iou": float(median_iou) if median_iou is not None else None,
            "iou_ge_070": float(iou_ge_070) if iou_ge_070 is not None else None,
            "iou_ge_085": float(iou_ge_085) if iou_ge_085 is not None else None,
        }

    return summary


def _update_summaries(run_dir, summary):
    for name in ("test_metrics.json", "experiment_summary.json"):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["per_class_metrics"] = summary
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse JSON: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill class-wise IoU diagnostics for runs with COCO ann/pred files."
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-update-summaries", action="store_true")
    args = parser.parse_args()

    topk = None if args.topk < 0 else args.topk

    run_dirs = _find_run_dirs(args.runs_dir)
    if not run_dirs:
        print(f"No runs with test_ann.json + test_preds.json under {args.runs_dir}")
        return

    for run_dir in run_dirs:
        out_path = os.path.join(run_dir, "classwise_metrics.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] {run_dir} (classwise_metrics.json exists)")
            continue

        ann_path, pred_path = _resolve_ann_pred_paths(run_dir)
        if not ann_path or not pred_path:
            print(f"[SKIP] {run_dir} (missing ann/pred)")
            continue

        try:
            summary = compute_classwise_from_coco(
                ann_path,
                pred_path,
                iou_thresh=args.iou_thresh,
                score_thresh=args.score_thresh,
                topk=topk,
            )
        except json.JSONDecodeError:
            print(f"[WARN] Failed to parse JSON in {run_dir}")
            continue

        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        if not args.no_update_summaries:
            _update_summaries(run_dir, summary)

        print(f"[OK] {run_dir}")


if __name__ == "__main__":
    main()
