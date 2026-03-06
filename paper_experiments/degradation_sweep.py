#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_experiments.common import (
    DEFAULT_PER_CLASS_CAP,
    compute_ap_for_image,
    extract_embedding,
    infer_predictions,
    load_gt_from_graph,
    load_image,
    load_label_maps,
    load_model,
    load_test_dirs,
    match_greedy_by_class,
    safe_makedirs,
)
from utils.geometry import clamp_bbox_xywh


def _build_roi_mask(graph_path: str, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    if not os.path.exists(graph_path):
        return mask
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for node in data.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue
        clamped = clamp_bbox_xywh(bbox, width, height)
        if clamped is None:
            continue
        x, y, w, h = clamped
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(min(width, x + w)), int(min(height, y + h))
        mask[y1:y2, x1:x2] = 1.0
    if mask.sum() == 0:
        mask[:, :] = 1.0
    return mask


def _apply_mask(pil_img: Image.Image, mask: np.ndarray, keep: bool) -> Image.Image:
    arr = np.array(pil_img).astype(np.float32)
    mask3 = np.stack([mask] * 3, axis=-1)
    if not keep:
        mask3 = 1.0 - mask3
    masked = arr * mask3
    return Image.fromarray(np.clip(masked, 0, 255).astype(np.uint8))


def _estimate_contrast(img: Image.Image) -> float:
    arr = np.array(img).astype(np.float32)
    return float(arr.std())


def degrade_blur(img: Image.Image, severity: float, params: Dict[str, float], **_) -> Tuple[Image.Image, Dict[str, float]]:
    sigma = severity * params.get("max_sigma", 4.5)
    if sigma < 1e-3:
        return img.copy(), {"sigma": 0.0}
    return img.filter(ImageFilter.GaussianBlur(radius=sigma)), {"sigma": float(sigma)}


def degrade_thicken(img: Image.Image, severity: float, params: Dict[str, float], **_) -> Tuple[Image.Image, Dict[str, float]]:
    scale = params.get("max_kernel", 7)
    kernel = 1 + int(np.round(severity * scale))
    if kernel <= 1:
        return img.copy(), {"kernel": 1}
    if kernel % 2 == 0:
        kernel += 1
    return img.filter(ImageFilter.MaxFilter(size=kernel)), {"kernel": kernel}


def degrade_clutter(
    img: Image.Image, severity: float, params: Dict[str, float], rng: np.random.Generator, **_
) -> Tuple[Image.Image, Dict[str, float]]:
    if severity < 1e-3:
        return img.copy(), {"rectangles": 0}
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    num_rect = max(1, int(np.round(severity * params.get("max_rectangles", 12))))
    width, height = img.size
    for _ in range(num_rect):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        w = int(rng.integers(max(10, width // 12), max(20, width // 5)))
        h = int(rng.integers(max(10, height // 12), max(20, height // 5)))
        x2 = min(width, x1 + w)
        y2 = min(height, y1 + h)
        color = tuple(int(x) for x in rng.integers(0, 256, size=3))
        alpha = int(np.clip(40 + severity * 120, 0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color + (alpha,))
    combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return combined, {"rectangles": num_rect}


DEGRADERS = {
    "blur": (degrade_blur, {"max_sigma": 5.5}),
    "stroke": (degrade_thicken, {"max_kernel": 9}),
    "clutter": (degrade_clutter, {"max_rectangles": 14}),
}


def apply_degradation(
    img: Image.Image,
    factor: str,
    severity: float,
    rng: np.random.Generator,
) -> Tuple[Image.Image, Dict[str, float]]:
    fn, params = DEGRADERS[factor]
    return fn(img, severity, params, rng=rng)


def collect_predictions(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    id2label: Dict[int, str],
    **infer_kwargs,
) -> Tuple[List, List, List]:
    boxes, scores, labels = infer_predictions(
        model,
        processor,
        image,
        device,
        id2label,
        **infer_kwargs,
    )
    return boxes, scores, labels


def summarize_factor(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for rec in results:
        factor = rec["factor"]
        summary.setdefault(factor, {"delta_recall85": [], "dist_l2": []})
        summary[factor]["delta_recall85"].append(rec["delta_recall85"])
        summary[factor]["dist_l2"].append(rec["dist_l2"])
    out: Dict[str, Dict[str, float]] = {}
    for factor, metric_map in summary.items():
        out[factor] = {
            "mean_delta_recall85": float(np.mean(metric_map["delta_recall85"])),
            "mean_dist_l2": float(np.mean(metric_map["dist_l2"])),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Controlled degradation sweep + distance mediation.")
    parser.add_argument("--ckpt", required=True, help="RT-DETR checkpoint for inference.")
    parser.add_argument("--out-dir", default="paper_experiments/out/degradation", help="Where to store results.")
    parser.add_argument("--image", default="model_baked.png", help="CAD image name to degrade.")
    parser.add_argument("--test-txt", default=None, help="Optional TEST_TXT override.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of layouts.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--max-layouts", type=int, default=50, help="Number of layouts to sweep (per factor).")
    parser.add_argument("--severity-steps", type=int, default=3, help="Severity levels per factor.")
    parser.add_argument("--min-severity", type=float, default=0.2, help="Minimum severity fraction.")
    args = parser.parse_args()

    safe_makedirs(args.out_dir)
    device = torch.device(args.device)
    level2, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(args.ckpt, device)

    test_dirs = load_test_dirs(args.test_txt)
    if args.limit and args.limit > 0:
        test_dirs = test_dirs[: args.limit]
    if args.max_layouts > 0:
        test_dirs = test_dirs[: args.max_layouts]

    severity_levels = np.linspace(args.min_severity, 1.0, max(1, args.severity_steps))
    rng = np.random.default_rng(123)
    infer_kwargs = {
        "score_thresh": 0.0,
        "topk_pre": 0,
        "final_k": 0,
        "per_class_cap": DEFAULT_PER_CLASS_CAP.copy(),
        "use_per_class_thresh": False,
    }

    records = []
    start_time = time.time()

    for idx, folder in enumerate(tqdm(test_dirs, desc="Layouts")):
        img_path = os.path.join(folder, args.image)
        graph_path = os.path.join(folder, "graph.json")
        if not os.path.exists(img_path) or not os.path.exists(graph_path):
            continue
        img = load_image(img_path)
        gt_boxes, gt_labels, _ = load_gt_from_graph(
            graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
        )

        boxes_clean, scores_clean, labels_clean = collect_predictions(
            model,
            processor,
            img,
            device,
            id2label,
            **infer_kwargs,
        )
        recall85_clean = float(
            sum(i >= 0.85 for i in match_greedy_by_class(gt_boxes, gt_labels, boxes_clean.tolist(), scores_clean.tolist(), labels_clean.tolist(), iou_thr=0.5))
            ) / max(len(gt_labels), 1)
        ap85_clean = compute_ap_for_image(
            gt_boxes,
            gt_labels,
            boxes_clean.tolist(),
            scores_clean.tolist(),
            labels_clean.tolist(),
            iou_thr=0.85,
            class_ids=label2id.values(),
        )
        mask = _build_roi_mask(graph_path, img.size[0], img.size[1])
        roi_clean = _apply_mask(img, mask, keep=True)
        bg_clean = _apply_mask(img, mask, keep=False)
        emb_clean = extract_embedding(model, processor, img, device)
        emb_roi_clean = extract_embedding(model, processor, roi_clean, device)
        emb_bg_clean = extract_embedding(model, processor, bg_clean, device)

        contrast_clean = _estimate_contrast(img)

        for factor in DEGRADERS:
            for severity in severity_levels:
                degraded, details = apply_degradation(img, factor, float(severity), rng)
                boxes_deg, scores_deg, labels_deg = collect_predictions(
                    model,
                    processor,
                    degraded,
                    device,
                    id2label,
                    **infer_kwargs,
                )

                recall85_deg = float(
                    sum(i >= 0.85 for i in match_greedy_by_class(gt_boxes, gt_labels, boxes_deg.tolist(), scores_deg.tolist(), labels_deg.tolist(), iou_thr=0.5))
                    ) / max(len(gt_labels), 1)
                ap85_deg = compute_ap_for_image(
                    gt_boxes,
                    gt_labels,
                    boxes_deg.tolist(),
                    scores_deg.tolist(),
                    labels_deg.tolist(),
                    iou_thr=0.85,
                    class_ids=label2id.values(),
                )

                emb_deg = extract_embedding(model, processor, degraded, device)
                roi_deg = _apply_mask(degraded, mask, keep=True)
                bg_deg = _apply_mask(degraded, mask, keep=False)
                emb_roi_deg = extract_embedding(model, processor, roi_deg, device)
                emb_bg_deg = extract_embedding(model, processor, bg_deg, device)

                dist_l2 = float(np.linalg.norm(emb_clean - emb_deg))
                dist_roi = float(np.linalg.norm(emb_roi_clean - emb_roi_deg))
                dist_bg = float(np.linalg.norm(emb_bg_clean - emb_bg_deg))

                mean_score = float(np.mean(scores_deg)) if scores_deg is not None and len(scores_deg) else 0.0
                n_detections = int(len(scores_deg)) if scores_deg is not None else 0
                contrast_deg = _estimate_contrast(degraded)

                records.append(
                    {
                        "layout_id": os.path.basename(folder.rstrip("/")),
                        "factor": factor,
                        "severity": float(severity),
                        "dist_l2": dist_l2,
                        "dist_roi": dist_roi,
                        "dist_bg": dist_bg,
                        "recall85_clean": recall85_clean,
                        "recall85_deg": recall85_deg,
                        "delta_recall85": recall85_clean - recall85_deg,
                        "ap85_clean": float(ap85_clean),
                        "ap85_deg": float(ap85_deg),
                        "delta_ap85": float(ap85_clean - ap85_deg),
                        "mean_score_deg": mean_score,
                        "n_detections_deg": n_detections,
                        "contrast_clean": contrast_clean,
                        "contrast_deg": contrast_deg,
                        "factor_params": details,
                    }
                )
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[progress] {idx + 1}/{len(test_dirs)} layouts, avg {elapsed / float(idx + 1):.2f}s/layout")

    if not records:
        raise RuntimeError("No degradation samples collected.")

    results_path = os.path.join(args.out_dir, "degradation_sweep.json")
    summary_path = os.path.join(args.out_dir, "degradation_summary.json")
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    summary = summarize_factor(records)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {len(records)} degraded samples to {results_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
