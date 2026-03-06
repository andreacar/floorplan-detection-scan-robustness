#!/usr/bin/env python3
import os
import sys
import json
import argparse
import yaml
from collections import defaultdict
from contextlib import redirect_stdout
import io

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, RTDetrForObjectDetection
from pycocotools.cocoeval import COCOeval

from utils.paths import load_split_list
from data.dataset import GraphRTDetrDataset
from data.collate import make_collate_fn
from data.coco_utils import build_coco_groundtruth, rtdetr_to_coco_json
from eval.coco_eval import coco_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

cfg = yaml.safe_load(open(args.config))

import config as config_module
from config import *  # noqa


# ===============================================================
# MODELS (A and B)
# ===============================================================
MODEL_A_DIR = cfg["model_a_dir"]
MODEL_B_DIR = cfg["model_b_dir"]

MODELS = {
    "ModelA_clean_trained": MODEL_A_DIR,
    "ModelB_scan_trained": MODEL_B_DIR,
}

# ===============================================================
# TEST VARIANTS (C and D)
# ===============================================================
IMAGE_VARIANTS = [
    "four_final_variants/03_scan_inside_boxes.png",           # C
    "four_final_variants/04_svg_clean_plus_scan_outside.png", # D
]

OUT_DIR = "./AB_on_CD_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================================================
# F1 SETTINGS (single operating point, not AP integration)
# ===============================================================
F1_IOU_THRESH = 0.50
F1_SCORE_THRESH = 0.05  # keep aligned with your diagnostic functions

# "Popular" / practically important subset
POPULAR_CLASSES = ["WALL", "DOOR", "WINDOW"]

# Optional: if you want importance-weighted F1 (NOT support-weighted).
# Leave empty {} to disable.
CLASS_IMPORTANCE_WEIGHTS = {
    # "WALL": 3.0,
    # "DOOR": 2.0,
    # "WINDOW": 2.0,
}


# ===============================================================
# PER-CLASS METRICS FROM COCOEVAL
# ===============================================================
def extract_per_class_ap_and_recall(cocoEval, id2label):
    """
    Returns:
        per_class_ap[class] = AP@[.50:.95]
        per_class_recall[class] = AR@100 @[.50:.95]
    """
    per_class_ap = {}
    per_class_recall = {}

    precision = cocoEval.eval["precision"]  # [T, R, K, A, M]
    recall = cocoEval.eval["recall"]        # [T, K, A, M]

    for k, class_name in id2label.items():
        # AP over IoU=.50:.95, area=all (0), maxDets=last
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        per_class_ap[class_name] = float(p.mean()) if p.size else 0.0

        # Recall over IoU=.50:.95, area=all (0), maxDets=last
        r = recall[:, k, 0, -1]
        r = r[r > -1]
        per_class_recall[class_name] = float(r.mean()) if r.size else 0.0

    return per_class_ap, per_class_recall


def coco_stats_for_catIds(cocoEval, catIds, verbose=False):
    """
    Run COCOeval again but restricted to catIds (subset of classes).
    Returns eval.stats (12 numbers).
    """
    e = COCOeval(cocoEval.cocoGt, cocoEval.cocoDt, "bbox")
    e.params.maxDets = cocoEval.params.maxDets
    e.params.catIds = list(catIds)

    if verbose:
        e.evaluate()
        e.accumulate()
        e.summarize()
    else:
        # silence COCOeval printing
        buf = io.StringIO()
        with redirect_stdout(buf):
            e.evaluate()
            e.accumulate()
            e.summarize()

    return [float(x) for x in e.stats]


# ===============================================================
# SIMPLE DETECTION F1 (TP/FP/FN) AT FIXED IOU + SCORE THRESH
# ===============================================================
def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_f1_from_coco_json(
    anno_json_path,
    pred_json_path,
    id2label,
    *,
    iou_thresh=0.5,
    score_thresh=0.05,
    include_class_names=None,
):
    """
    Greedy matching per class, per image, sorted by prediction score (VOC-style).

    Returns:
        {
          "per_class": {cls: {"tp":..,"fp":..,"fn":..,"support":..,"precision":..,"recall":..,"f1":..}},
          "micro": {...},
          "macro_f1": ...,
          "weighted_f1_support": ...,
          "weighted_f1_importance": ... (if CLASS_IMPORTANCE_WEIGHTS is non-empty)
        }
    """
    with open(anno_json_path, "r") as f:
        gt = json.load(f)
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    # Determine which classes to include (by name)
    if include_class_names is None:
        include_class_names = list(id2label.values())
    include_class_names = set(include_class_names)

    # Build name<->id maps
    name_to_id = {name: cid for cid, name in id2label.items()}
    include_class_ids = {name_to_id[n] for n in include_class_names if n in name_to_id}

    # Group GT by (image_id, class_id)
    gt_by_img_cls = defaultdict(lambda: defaultdict(list))
    for ann in gt.get("annotations", []):
        cid = ann["category_id"]
        if cid not in include_class_ids:
            continue
        gt_by_img_cls[ann["image_id"]][cid].append(ann["bbox"])

    # Group preds by (image_id, class_id)
    pred_by_img_cls = defaultdict(lambda: defaultdict(list))
    for p in preds:
        cid = p["category_id"]
        if cid not in include_class_ids:
            continue
        if p.get("score", 0.0) < score_thresh:
            continue
        pred_by_img_cls[p["image_id"]][cid].append((float(p["score"]), p["bbox"]))

    # Accumulate counts
    per_class_counts = {
        id2label[cid]: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for cid in include_class_ids
    }

    all_image_ids = set(gt_by_img_cls.keys()) | set(pred_by_img_cls.keys())

    for img_id in all_image_ids:
        for cid in include_class_ids:
            gt_boxes = gt_by_img_cls[img_id].get(cid, [])
            pred_list = pred_by_img_cls[img_id].get(cid, [])
            pred_list.sort(key=lambda x: x[0], reverse=True)

            matched = [False] * len(gt_boxes)

            # support = number of GT instances
            per_class_counts[id2label[cid]]["support"] += len(gt_boxes)

            # greedy match predictions to unmatched GT
            for _, pb in pred_list:
                best_iou = 0.0
                best_j = -1
                for j, gb in enumerate(gt_boxes):
                    if matched[j]:
                        continue
                    iou = _iou_xywh(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_thresh and best_j >= 0:
                    matched[best_j] = True
                    per_class_counts[id2label[cid]]["tp"] += 1
                else:
                    per_class_counts[id2label[cid]]["fp"] += 1

            # remaining GT are FN
            per_class_counts[id2label[cid]]["fn"] += sum(1 for m in matched if not m)

    # Convert counts -> metrics
    per_class = {}
    tp_all = fp_all = fn_all = 0
    f1_list = []
    support_total = 0

    # For optional importance-weighted F1
    imp_weight_sum = 0.0
    imp_weighted_f1_sum = 0.0

    for cls, c in per_class_counts.items():
        tp, fp, fn, sup = c["tp"], c["fp"], c["fn"], c["support"]
        tp_all += tp
        fp_all += fp
        fn_all += fn
        support_total += sup

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "support": int(sup),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

        f1_list.append(f1)

        if CLASS_IMPORTANCE_WEIGHTS:
            w = float(CLASS_IMPORTANCE_WEIGHTS.get(cls, 0.0))
            if w > 0:
                imp_weight_sum += w
                imp_weighted_f1_sum += w * f1

    # micro
    micro_prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    micro_rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

    # macro
    macro_f1 = float(sum(f1_list) / len(f1_list)) if f1_list else 0.0

    # support-weighted
    weighted_support_f1 = 0.0
    if support_total > 0:
        weighted_support_f1 = sum(per_class[cls]["f1"] * per_class[cls]["support"] for cls in per_class) / support_total

    # importance-weighted
    weighted_importance_f1 = None
    if CLASS_IMPORTANCE_WEIGHTS and imp_weight_sum > 0:
        weighted_importance_f1 = imp_weighted_f1_sum / imp_weight_sum

    out = {
        "per_class": per_class,
        "micro": {
            "tp": int(tp_all),
            "fp": int(fp_all),
            "fn": int(fn_all),
            "precision": float(micro_prec),
            "recall": float(micro_rec),
            "f1": float(micro_f1),
        },
        "macro_f1": float(macro_f1),
        "weighted_f1_support": float(weighted_support_f1),
        "weighted_f1_importance": float(weighted_importance_f1) if weighted_importance_f1 is not None else None,
        "settings": {
            "iou_thresh": float(iou_thresh),
            "score_thresh": float(score_thresh),
            "classes": sorted(list(include_class_names)),
        },
    }
    return out


# ===============================================================
# EVALUATION CORE
# ===============================================================
def evaluate_model(model_name, model_dir, image_variant):
    print(f"\n=== Evaluating {model_name} on {image_variant} ===")

    config_module.IMAGE_FILENAME = image_variant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(BACKBONE)

    test_dirs = load_split_list(TEST_TXT)
    ds_test = GraphRTDetrDataset(test_dirs, processor, "hierarchy_config.py", augment=False)

    collate = make_collate_fn(ds_test.image_processor)
    loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = RTDetrForObjectDetection.from_pretrained(model_dir).to(device)
    model.eval()

    pred_json = os.path.join(OUT_DIR, f"{model_name}_{os.path.basename(image_variant)}_pred.json")
    anno_json = os.path.join(OUT_DIR, f"{model_name}_{os.path.basename(image_variant)}_gt.json")

    build_coco_groundtruth(ds_test, anno_json)
    rtdetr_to_coco_json(model, ds_test, device, pred_json)

    # COCO overall + cocoEval handle
    stats, cocoEval = coco_evaluate(ds_test, anno_json, pred_json, return_coco=True)

    # per-class AP/AR from COCOeval tensors
    per_class_ap, per_class_recall = extract_per_class_ap_and_recall(cocoEval, ds_test.id2label)

    # F1 metrics (all classes)
    f1_all = compute_f1_from_coco_json(
        anno_json,
        pred_json,
        ds_test.id2label,
        iou_thresh=F1_IOU_THRESH,
        score_thresh=F1_SCORE_THRESH,
        include_class_names=None,
    )

    # F1 metrics (popular subset)
    f1_pop = compute_f1_from_coco_json(
        anno_json,
        pred_json,
        ds_test.id2label,
        iou_thresh=F1_IOU_THRESH,
        score_thresh=F1_SCORE_THRESH,
        include_class_names=POPULAR_CLASSES,
    )

    # COCO restricted to popular classes (optional but very useful for the paper)
    popular_catIds = [ds_test.label2id[c] for c in POPULAR_CLASSES if c in ds_test.label2id]
    coco_popular = None
    if len(popular_catIds) > 0:
        coco_pop_stats = coco_stats_for_catIds(cocoEval, popular_catIds, verbose=False)
        coco_popular = {
            "AP": coco_pop_stats[0],
            "AP50": coco_pop_stats[1],
            "AP75": coco_pop_stats[2],
            "AP_small": coco_pop_stats[3],
            "AP_medium": coco_pop_stats[4],
            "AP_large": coco_pop_stats[5],
            "AR_1": coco_pop_stats[6],
            "AR_10": coco_pop_stats[7],
            "AR_100": coco_pop_stats[8],
        }

    summary = {
        "model": model_name,
        "image_variant": image_variant,
        "overall": {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
            "AR_1": float(stats[6]),
            "AR_10": float(stats[7]),
            "AR_100": float(stats[8]),
        },
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_recall,
        "f1_all_classes": f1_all,
        "f1_popular_classes": f1_pop,
        "coco_popular_classes": coco_popular,
        "popular_classes": POPULAR_CLASSES,
    }

    out_path = os.path.join(
        OUT_DIR,
        f"{model_name}_{os.path.basename(image_variant).replace('.png','')}_summary.json",
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved to:", out_path)

    # quick console sanity
    print(f"F1(all) micro={summary['f1_all_classes']['micro']['f1']:.3f} | "
          f"weighted={summary['f1_all_classes']['weighted_f1_support']:.3f} | "
          f"macro={summary['f1_all_classes']['macro_f1']:.3f}")
    print(f"F1(popular={POPULAR_CLASSES}) micro={summary['f1_popular_classes']['micro']['f1']:.3f} | "
          f"weighted={summary['f1_popular_classes']['weighted_f1_support']:.3f} | "
          f"macro={summary['f1_popular_classes']['macro_f1']:.3f}")

    if coco_popular is not None:
        print(f"COCO(popular) AP={coco_popular['AP']:.3f} | AP50={coco_popular['AP50']:.3f} | AP75={coco_popular['AP75']:.3f}")

    return summary


# ===============================================================
# MAIN
# ===============================================================
def main():
    all_results = []

    for image_variant in IMAGE_VARIANTS:
        for model_name, model_dir in MODELS.items():
            res = evaluate_model(model_name, model_dir, image_variant)
            all_results.append(res)

    with open(os.path.join(OUT_DIR, "ALL_AB_ON_CD_RESULTS.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== DONE: A and B tested on C and D (COCO + per-class + F1 + popular subset) ===")
    print("Results saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
