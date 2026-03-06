#!/usr/bin/env python3
import os
import sys
import json
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, RTDetrForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.paths import load_split_list
from data.dataset import GraphRTDetrDataset
from data.collate import make_collate_fn
from data.coco_utils import build_coco_groundtruth, rtdetr_to_coco_json

import config as config_module
from config import *  # BACKBONE, TEST_TXT, BATCH_SIZE, COCO_MAX_DETS, ...


# ===============================================================
# MODELS (A and B)
# ===============================================================
MODELS = {}

# ===============================================================
# TEST VARIANTS (C and D)
# ===============================================================
DEFAULT_IMAGE_VARIANTS = [
    "four_final_variants/03_scan_inside_boxes.png",
    "four_final_variants/04_svg_clean_plus_scan_outside.png",
]

IMAGE_VARIANTS = list(DEFAULT_IMAGE_VARIANTS)
OUT_DIR = "./AB_on_CD_results"

# ===============================================================
# F1 SETTINGS
# ===============================================================
F1_IOU_THRESH = 0.50
F1_SCORE_THRESH = 0.05
F1_MAX_DETS = 100

POPULAR_TOPK = 3
CLASS_IMPORTANCE_WEIGHTS = None


# ===============================================================
# COCO EVAL HELPERS
# ===============================================================
def _load_dets_filtered(pred_json_path, score_thresh=None):
    with open(pred_json_path, "r") as f:
        dets = json.load(f)
    if score_thresh is None:
        return dets
    return [d for d in dets if float(d.get("score", 0.0)) >= float(score_thresh)]


def run_coco_eval(anno_json_path, pred_json_path, catIds=None, summarize=True, score_thresh_for_loading=None):
    coco_gt = COCO(anno_json_path)
    dets = _load_dets_filtered(pred_json_path, score_thresh_for_loading)
    coco_dt = coco_gt.loadRes(dets)

    cocoEval = COCOeval(coco_gt, coco_dt, "bbox")
    cocoEval.params.maxDets = COCO_MAX_DETS
    if catIds is not None:
        cocoEval.params.catIds = list(catIds)

    cocoEval.evaluate()
    cocoEval.accumulate()
    if summarize:
        cocoEval.summarize()

    return cocoEval.stats, cocoEval


def extract_per_class_ap_and_recall(cocoEval):
    coco_gt = cocoEval.cocoGt
    catIds = cocoEval.params.catIds
    precision = cocoEval.eval["precision"]
    recall = cocoEval.eval["recall"]

    per_class_ap = {}
    per_class_recall = {}

    for k_idx, catId in enumerate(catIds):
        name = coco_gt.cats[catId]["name"]

        p = precision[:, :, k_idx, 0, -1]
        p = p[p > -1]
        per_class_ap[name] = float(p.mean()) if p.size else 0.0

        r = recall[:, k_idx, 0, -1]
        r = r[r > -1]
        per_class_recall[name] = float(r.mean()) if r.size else 0.0

    return per_class_ap, per_class_recall


def pick_popular_classes_from_gt(anno_json_path, topk=3):
    coco_gt = COCO(anno_json_path)
    counts = defaultdict(int)
    for ann in coco_gt.anns.values():
        counts[ann["category_id"]] += 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    popular_catIds = [cid for cid, _ in ranked[:topk]]
    popular_names = [coco_gt.cats[cid]["name"] for cid in popular_catIds]
    return popular_names, popular_catIds


# ===============================================================
# F1 FROM COCO MATCHING
# ===============================================================
def f1_from_coco_matching(
    anno_json_path,
    pred_json_path,
    catIds,
    iou_thresh=0.5,
    score_thresh=0.05,
    max_dets=100,
    importance_weights=None,
):
    coco_gt = COCO(anno_json_path)
    dets = _load_dets_filtered(pred_json_path, score_thresh)
    coco_dt = coco_gt.loadRes(dets)

    cocoEval = COCOeval(coco_gt, coco_dt, "bbox")
    cocoEval.params.catIds = list(catIds)
    cocoEval.params.iouThrs = np.array([float(iou_thresh)])
    cocoEval.params.maxDets = [int(max_dets)]

    cocoEval.evaluate()
    cocoEval.accumulate()

    per_class = {}
    tp_sum = fp_sum = fn_sum = 0

    for eval_img in cocoEval.evalImgs:
        if eval_img is None:
            continue

        catId = eval_img["category_id"]
        if catId not in catIds:
            continue

        name = coco_gt.cats[catId]["name"]

        dtMatches = np.array(eval_img["dtMatches"])[0]
        dtIgnore = np.array(eval_img["dtIgnore"])[0]
        gtMatches = np.array(eval_img["gtMatches"])[0]
        gtIgnore = np.array(eval_img["gtIgnore"])

        tp = int(np.sum((dtMatches > 0) & (dtIgnore == 0)))
        fp = int(np.sum((dtMatches == 0) & (dtIgnore == 0)))
        fn = int(np.sum((gtMatches == 0) & (gtIgnore == 0)))
        support = int(np.sum(gtIgnore == 0))

        if name not in per_class:
            per_class[name] = {"tp": 0, "fp": 0, "fn": 0, "support": 0}

        per_class[name]["tp"] += tp
        per_class[name]["fp"] += fp
        per_class[name]["fn"] += fn
        per_class[name]["support"] += support

    f1s, supports = [], []
    for d in per_class.values():
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        d["precision"] = prec
        d["recall"] = rec
        d["f1"] = (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
        f1s.append(d["f1"])
        supports.append(d["support"])

        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    micro_prec = tp_sum / (tp_sum + fp_sum) if tp_sum + fp_sum > 0 else 0.0
    micro_rec = tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if micro_prec + micro_rec > 0 else 0.0

    weighted_f1_support = (
        float(np.sum(np.array(f1s) * np.array(supports)) / np.sum(supports))
        if np.sum(supports) > 0
        else 0.0
    )

    return {
        "per_class": per_class,
        "micro": {
            "precision": micro_prec,
            "recall": micro_rec,
            "f1": micro_f1,
        },
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_f1_support": weighted_f1_support,
    }


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
    _ = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = RTDetrForObjectDetection.from_pretrained(model_dir).to(device)
    model.eval()

    pred_json = os.path.join(OUT_DIR, f"{model_name}_{os.path.basename(image_variant)}_pred.json")
    anno_json = os.path.join(OUT_DIR, f"{model_name}_{os.path.basename(image_variant)}_gt.json")

    build_coco_groundtruth(ds_test, anno_json)
    rtdetr_to_coco_json(model, ds_test, device, pred_json)

    popular_names, popular_catIds = pick_popular_classes_from_gt(anno_json, topk=POPULAR_TOPK)

    stats, cocoEval = run_coco_eval(anno_json, pred_json, catIds=None, summarize=True)
    per_class_ap, per_class_recall = extract_per_class_ap_and_recall(cocoEval)

    coco_gt_tmp = COCO(anno_json)
    all_catIds = sorted(coco_gt_tmp.cats.keys())

    f1_all = f1_from_coco_matching(
        anno_json, pred_json, all_catIds, F1_IOU_THRESH, F1_SCORE_THRESH, F1_MAX_DETS
    )

    f1_pop = f1_from_coco_matching(
        anno_json, pred_json, popular_catIds, F1_IOU_THRESH, F1_SCORE_THRESH, F1_MAX_DETS
    )

    stats_pop, _ = run_coco_eval(
        anno_json, pred_json, catIds=popular_catIds, summarize=False
    )

    if stats_pop is None or len(stats_pop) < 9:
        coco_popular = {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_1": 0.0,
            "AR_10": 0.0,
            "AR_100": 0.0,
            "note": "No valid GT/detections for popular classes",
        }
    else:
        coco_popular = {
            "AP": float(stats_pop[0]),
            "AP50": float(stats_pop[1]),
            "AP75": float(stats_pop[2]),
            "AP_small": float(stats_pop[3]),
            "AP_medium": float(stats_pop[4]),
            "AP_large": float(stats_pop[5]),
            "AR_1": float(stats_pop[6]),
            "AR_10": float(stats_pop[7]),
            "AR_100": float(stats_pop[8]),
        }

    summary = {
        "model": model_name,
        "image_variant": image_variant,
        "overall": {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AR_100": float(stats[8]),
        },
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_recall,
        "f1_all_classes": f1_all,
        "f1_popular_classes": f1_pop,
        "coco_popular_classes": coco_popular,
        "popular_classes": popular_names,
    }

    out_path = os.path.join(
        OUT_DIR,
        f"{model_name}_{os.path.basename(image_variant).replace('.png','')}_summary.json",
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved to:", out_path)
    return summary


# ===============================================================
# MAIN
# ===============================================================
def main():
    global MODELS, IMAGE_VARIANTS, OUT_DIR

    parser = argparse.ArgumentParser(description="Evaluate A/B models on C/D variants.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config with keys: model_a_dir, model_b_dir.",
    )
    parser.add_argument(
        "--model-a-dir",
        type=str,
        default="",
        help="HF checkpoint directory for model A (clean-trained).",
    )
    parser.add_argument(
        "--model-b-dir",
        type=str,
        default="",
        help="HF checkpoint directory for model B (scan-trained).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR,
        help="Output directory for JSON + COCO artifacts.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Image variant filenames to evaluate (defaults to C/D).",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config:
        import yaml  # optional dependency; only needed when using --config
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    model_a_dir = args.model_a_dir or cfg.get("model_a_dir", "")
    model_b_dir = args.model_b_dir or cfg.get("model_b_dir", "")
    if not model_a_dir or not model_b_dir:
        raise SystemExit(
            "Missing model dirs. Provide either:\n"
            "- --config <yaml> with model_a_dir/model_b_dir, or\n"
            "- --model-a-dir and --model-b-dir."
        )

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    IMAGE_VARIANTS = list(args.variants) if args.variants else list(DEFAULT_IMAGE_VARIANTS)
    MODELS = {
        "ModelA_clean_trained": model_a_dir,
        "ModelB_scan_trained": model_b_dir,
    }

    all_results = []
    for image_variant in IMAGE_VARIANTS:
        for model_name, model_dir in MODELS.items():
            all_results.append(evaluate_model(model_name, model_dir, image_variant))

    with open(os.path.join(OUT_DIR, "ALL_AB_ON_CD_RESULTS.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== DONE: A and B tested on C and D ===")


if __name__ == "__main__":
    main()
