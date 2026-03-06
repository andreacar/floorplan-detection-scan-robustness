#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
from typing import List, Dict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _epoch_path(metrics_dir: str, split: str, epoch: int, kind: str) -> str:
    if split == "val":
        return os.path.join(metrics_dir, f"val_{kind}_epoch_{epoch:03d}.json")
    if split == "test":
        return os.path.join(metrics_dir, f"test_{kind}.json")
    raise ValueError(f"Unknown split: {split}")


def _load_coco(ann_path: str, pred_path: str) -> COCOeval:
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Missing ann file: {ann_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing pred file: {pred_path}")
    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(pred_path)
    return coco_gt, coco_dt


def _get_cat_ids(coco_gt: COCO, class_names: List[str]) -> List[int]:
    name_to_id = {c["name"]: c["id"] for c in coco_gt.dataset.get("categories", [])}
    missing = [c for c in class_names if c not in name_to_id]
    if missing:
        raise ValueError(f"Classes not found in categories: {missing}")
    return [name_to_id[c] for c in class_names]


def _per_class_ap(eval_obj: COCOeval, class_names: List[str], cat_ids: List[int]) -> Dict[str, Dict[str, float]]:
    precision = eval_obj.eval["precision"]  # [T, R, K, A, M]
    iou_thrs = eval_obj.params.iouThrs
    ap_by_class = {}
    for k, (name, cid) in enumerate(zip(class_names, cat_ids)):
        ap_all = precision[:, :, k, 0, -1]
        ap_all = ap_all[ap_all > -1]
        ap = float(np.mean(ap_all)) if ap_all.size else float("nan")

        # AP50/AP75 for the class
        ap50 = float("nan")
        ap75 = float("nan")
        if iou_thrs is not None:
            for thr, label in [(0.50, "ap50"), (0.75, "ap75")]:
                idx = int(np.where(np.isclose(iou_thrs, thr))[0][0])
                vals = precision[idx, :, k, 0, -1]
                vals = vals[vals > -1]
                score = float(np.mean(vals)) if vals.size else float("nan")
                if label == "ap50":
                    ap50 = score
                else:
                    ap75 = score

        ap_by_class[name] = {"ap": ap, "ap50": ap50, "ap75": ap75}
    return ap_by_class


def main():
    parser = argparse.ArgumentParser(
        description="Compute COCO stats for a class subset (e.g., WALL/WINDOW/DOOR) at a given epoch."
    )
    parser.add_argument("--run-dir", required=True, help="Run dir containing metrics/ (e.g. exp2_scanned)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--epoch", type=int, default=1, help="Epoch number for val split (ignored for test).")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["WALL", "WINDOW", "DOOR"],
        help="Class names to include (space-separated).",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    metrics_dir = os.path.join(args.run_dir, "metrics")
    ann_path = _epoch_path(metrics_dir, args.split, args.epoch, "ann")
    pred_path = _epoch_path(metrics_dir, args.split, args.epoch, "preds")

    coco_gt, coco_dt = _load_coco(ann_path, pred_path)
    cat_ids = _get_cat_ids(coco_gt, args.classes)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = cat_ids
    coco_eval.params.imgIds = coco_gt.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "run_dir": args.run_dir,
        "split": args.split,
        "epoch": args.epoch if args.split == "val" else None,
        "classes": args.classes,
        "cat_ids": cat_ids,
        "coco_stats": {
            "AP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
            "AP75": float(coco_eval.stats[2]),
            "AP_small": float(coco_eval.stats[3]),
            "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5]),
            "AR_1": float(coco_eval.stats[6]),
            "AR_10": float(coco_eval.stats[7]),
            "AR_100": float(coco_eval.stats[8]),
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        },
        "per_class_ap": _per_class_ap(coco_eval, args.classes, cat_ids),
    }

    print("\n=== Subset COCO stats (classes: {}) ===".format(", ".join(args.classes)))
    for k, v in stats["coco_stats"].items():
        print(f"{k:>9}: {v:.4f}")

    print("\n=== Per-class AP ===")
    for cls, vals in stats["per_class_ap"].items():
        print(f"{cls:>7}: AP={vals['ap']:.4f}  AP50={vals['ap50']:.4f}  AP75={vals['ap75']:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved JSON: {args.out}")


if __name__ == "__main__":
    main()
