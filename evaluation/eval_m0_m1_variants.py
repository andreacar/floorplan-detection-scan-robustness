#!/usr/bin/env python3
"""
Evaluate M0 vs M1 on CAD / Scan / C (scan-geometry) variants.

Reports:
- COCO: AP, AP50, AP75, AP_S
- Error decomposition: missed / loose / tight fractions
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrForObjectDetection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.paths import load_split_list
from data.dataset import GraphRTDetrDataset
from data.coco_utils import build_coco_groundtruth, rtdetr_to_coco_json
from eval.coco_eval import coco_evaluate
from paper_experiments.common import (
    infer_predictions,
    load_gt_from_graph,
    match_greedy_by_class,
)
import config as config_module
from config import TEST_TXT


def _load_summary(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_ckpt_path(spec: str, exp_name: str = "exp1_clean") -> str:
    if os.path.isdir(spec):
        cfg = os.path.join(spec, "config.json")
        if os.path.exists(cfg):
            return spec
        summary = os.path.join(spec, "ALL_EXPERIMENTS_SUMMARY.json")
        if os.path.exists(summary):
            data = _load_summary(summary)
            for exp in data.get("experiments", []):
                if exp.get("experiment") == exp_name and exp.get("best_dir"):
                    return exp["best_dir"]
        candidate = os.path.join(spec, exp_name, "checkpoints", "best")
        if os.path.exists(candidate):
            return candidate
    if os.path.isfile(spec):
        if os.path.basename(spec) == "ALL_EXPERIMENTS_SUMMARY.json":
            data = _load_summary(spec)
            for exp in data.get("experiments", []):
                if exp.get("experiment") == exp_name and exp.get("best_dir"):
                    return exp["best_dir"]
    return spec


def load_model(ckpt_path: str, device: torch.device):
    try:
        processor = AutoImageProcessor.from_pretrained(ckpt_path)
    except OSError:
        processor = AutoImageProcessor.from_pretrained(config_module.BACKBONE)
    model = RTDetrForObjectDetection.from_pretrained(ckpt_path).to(device)
    model.eval()
    return processor, model


def build_dataset(
    test_dirs: List[str],
    processor,
    image_name: str,
    hier_path: str,
):
    config_module.IMAGE_FILENAME = image_name
    ds = GraphRTDetrDataset(test_dirs, processor, hier_path, augment=False)
    return ds


def compute_decomposition(
    model,
    processor,
    dataset: GraphRTDetrDataset,
    device: torch.device,
    score_thresh: float,
    final_k: int,
    loose_iou: float = 0.5,
    tight_iou: float = 0.85,
) -> Dict[str, float]:
    overall = {"total": 0, "missed": 0, "loose": 0, "tight": 0}

    for img_path, graph_path, _ in dataset.samples:
        img = Image.open(img_path).convert("RGB")
        gt_boxes, gt_labels, _ = load_gt_from_graph(
            graph_path,
            img.size[0],
            img.size[1],
            dataset.map_raw_to_l2,
            dataset.label2id,
        )
        if not gt_boxes:
            continue

        boxes, scores, labels = infer_predictions(
            model,
            processor,
            img,
            device,
            dataset.id2label,
            score_thresh=score_thresh,
            final_k=final_k,
        )

        matched_iou = match_greedy_by_class(
            gt_boxes,
            gt_labels,
            boxes.tolist(),
            scores.tolist(),
            labels.tolist(),
            iou_thr=loose_iou,
        )

        for iou_val in matched_iou:
            overall["total"] += 1
            if iou_val < loose_iou:
                overall["missed"] += 1
            elif iou_val < tight_iou:
                overall["loose"] += 1
            else:
                overall["tight"] += 1

    total = max(overall["total"], 1)
    overall["missed_frac"] = overall["missed"] / total
    overall["loose_frac"] = overall["loose"] / total
    overall["tight_frac"] = overall["tight"] / total
    return overall


def coco_metrics(stats: List[float]) -> Dict[str, float]:
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_S": float(stats[3]),
    }


def evaluate_variant(
    model_name: str,
    model,
    processor,
    dataset: GraphRTDetrDataset,
    device: torch.device,
    out_dir: str,
    variant_key: str,
    score_thresh: float,
    final_k: int,
) -> Dict[str, object]:
    dataset.image_processor = processor

    ann_path = os.path.join(out_dir, f"{variant_key}_gt.json")
    if not os.path.exists(ann_path):
        build_coco_groundtruth(dataset, ann_path)

    pred_path = os.path.join(out_dir, f"{variant_key}_{model_name}_pred.json")
    rtdetr_to_coco_json(model, dataset, device, pred_path)
    stats = coco_evaluate(dataset, ann_path, pred_path)

    decomp = compute_decomposition(
        model,
        processor,
        dataset,
        device,
        score_thresh=score_thresh,
        final_k=final_k,
    )

    return {
        "coco_stats": [float(x) for x in stats],
        **coco_metrics(stats),
        **decomp,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate M0 vs M1 on CAD/Scan/C variants.")
    parser.add_argument("--m0", required=True, help="Path to M0 checkpoint dir (CAD baseline).")
    parser.add_argument("--m1", required=True, help="Path to M1 checkpoint dir (stroke-aug CAD).")
    parser.add_argument("--out-dir", default="RT_DETR_final/eval_m0_m1", help="Output directory.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto).")
    parser.add_argument("--cad", default="model_baked.png", help="CAD image filename.")
    parser.add_argument("--scan", default="F1_scaled.png", help="Scanned image filename.")
    parser.add_argument(
        "--variant-c",
        default="four_final_variants/03_scan_inside_boxes.png",
        help="C variant filename (scan geometry, clean background).",
    )
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Score threshold for decomposition.")
    parser.add_argument("--final-k", type=int, default=100, help="Max detections per image for decomposition.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    test_txt = args.test_txt or TEST_TXT
    test_dirs = load_split_list(test_txt)
    hier_path = os.path.join(PROJECT_ROOT, "hierarchy_config.py")

    m0_path = resolve_ckpt_path(args.m0)
    m1_path = resolve_ckpt_path(args.m1)
    if not os.path.isdir(m0_path):
        raise RuntimeError(f"M0 checkpoint path not found: {m0_path}")
    if not os.path.isdir(m1_path):
        raise RuntimeError(f"M1 checkpoint path not found: {m1_path}")

    processor_m0, model_m0 = load_model(m0_path, device)
    processor_m1, model_m1 = load_model(m1_path, device)

    variants: List[Tuple[str, str]] = [
        ("T1_scan", args.scan),
        ("T2_C", args.variant_c),
        ("T0_CAD", args.cad),
    ]

    results: Dict[str, object] = {
        "models": {"M0": m0_path, "M1": m1_path},
        "variants": {},
    }

    for key, image_name in variants:
        ds = build_dataset(test_dirs, processor_m0, image_name, hier_path)
        results["variants"][key] = {
            "image": image_name,
            "M0": evaluate_variant(
                "M0",
                model_m0,
                processor_m0,
                ds,
                device,
                args.out_dir,
                key,
                score_thresh=args.score_thresh,
                final_k=args.final_k,
            ),
            "M1": evaluate_variant(
                "M1",
                model_m1,
                processor_m1,
                ds,
                device,
                args.out_dir,
                key,
                score_thresh=args.score_thresh,
                final_k=args.final_k,
            ),
        }

        for name in ("M0", "M1"):
            res = results["variants"][key][name]
            print(
                f"[{key}] {name} AP={res['AP']:.4f} AP50={res['AP50']:.4f} "
                f"AP75={res['AP75']:.4f} AP_S={res['AP_S']:.4f} | "
                f"missed={res['missed_frac']:.3f} loose={res['loose_frac']:.3f} "
                f"tight={res['tight_frac']:.3f}"
            )

    out_path = os.path.join(args.out_dir, "m0_m1_summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
