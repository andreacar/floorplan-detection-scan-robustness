#!/usr/bin/env python3
"""
Best-checkpoint test visualization + deeper analytics for RT-DETR on GraphRTDetrDataset.

What this adds vs your current script:
- Strict (class-aware) metrics + localization-only metrics (class-agnostic matching)
- Confusion accounting (GT class matched by wrong predicted class at IoU>=thr)
- Per-class PR curves + AP@IoU for your chosen IoU threshold
- Score histograms for TP / FP / CONF, per class
- IoU histograms for TP + CONF, per class
- Size-bucket recall (COCO-ish small/medium/large) for strict + localization
- Worst-cases ranking (most FN / FP / CONF) dumped to JSON
- Visual overlay distinguishes TP / FP / FN / CONF clearly
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import json
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from transformers import AutoImageProcessor, RTDetrForObjectDetection

from utils.paths import load_split_list
from utils.geometry import clamp_bbox_xywh
from data.dataset import GraphRTDetrDataset, apply_subset
from config import CKPT_DIR, TEST_TXT, VIS_DIR, CLASS_COLORS, MAX_VIS, SUBSET_TEST


DEFAULT_PER_CLASS_CAP = {
    "WALL": 60,
    "DOOR": 30,
    "WINDOW": 30,
    "COLUMN": 20,
    "STAIR": 20,
    "RAILING": 20,
}

PER_CLASS_THRESH = {
    "WALL": 0.011,
    "COLUMN": 0.031,
    "STAIR": 0.040,
    "RAILING": 0.036,
    "DOOR": 0.015,
    "WINDOW": 0.043,
}


# -----------------------------
# Helpers / small utilities
# -----------------------------
def _parse_caps(caps_str: str) -> dict:
    caps = {}
    if not caps_str:
        return caps
    for part in caps_str.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k and v.isdigit():
            caps[k] = int(v)
    return caps


def _label_name(id2label: dict, lab_int: int) -> str:
    return id2label.get(lab_int, str(lab_int))


def _clip_and_filter_xyxy(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, w: int, h: int):
    """Clip xyxy boxes to image bounds and drop degenerate boxes."""
    if boxes.numel() == 0:
        return boxes, scores, labels

    b = boxes.clone()
    b[:, 0] = b[:, 0].clamp(0, w)
    b[:, 2] = b[:, 2].clamp(0, w)
    b[:, 1] = b[:, 1].clamp(0, h)
    b[:, 3] = b[:, 3].clamp(0, h)

    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep], scores[keep], labels[keep]


def _pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [N,4] xyxy, b: [M,4] xyxy
    returns [N,M]
    """
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)

    a = a.float()
    b = b.float()

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])         # [N,M,2]
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])         # [N,M,2]
    wh = (rb - lt).clamp(min=0)                            # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]                      # [N,M]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def _one_to_one_match_by_iou(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    iou_thr: float,
) -> Tuple[Dict[int, int], Dict[int, int], torch.Tensor]:
    """
    One-to-one matching maximizing IoU (greedy by IoU desc).
    Returns:
    gt_to_pred: {gi: pi}
    pred_to_gt: {pi: gi}
    ious: [G,P] pairwise IoU matrix
    """
    ious = _pairwise_iou_xyxy(gt_boxes, pred_boxes)
    if ious.numel() == 0:
        return {}, {}, ious

    pairs = torch.nonzero(ious >= float(iou_thr), as_tuple=False)  # [K,2]
    if pairs.numel() == 0:
        return {}, {}, ious

    pair_ious = ious[pairs[:, 0], pairs[:, 1]]
    order = torch.argsort(pair_ious, descending=True)

    gt_to_pred: Dict[int, int] = {}
    pred_to_gt: Dict[int, int] = {}

    for oi in order.tolist():
        gi = int(pairs[oi, 0].item())
        pi = int(pairs[oi, 1].item())
        if gi in gt_to_pred or pi in pred_to_gt:
            continue
        gt_to_pred[gi] = pi
        pred_to_gt[pi] = gi

    return gt_to_pred, pred_to_gt, ious


def _apply_final_policy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    id2label: dict,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    per_class_cap: dict,
    default_cap: int,
):
    """
    Shipping policy:
    - per-class score threshold
    - global topk_pre
    - per-class cap
    - final_k limit
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    # --- Per-class score threshold ---
    if score_thresh > 0.0:
        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        for i, (lab, sc) in enumerate(zip(labels, scores)):
            name = _label_name(id2label, int(lab.item()))
            thr = PER_CLASS_THRESH.get(name, score_thresh)
            if sc >= thr:
                keep_mask[i] = True

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]

        if boxes.numel() == 0:
            return boxes, scores, labels

    # --- Pre top-K by score (global) ---
    if topk_pre and scores.numel() > int(topk_pre):
        keep = torch.argsort(scores, descending=True)[: int(topk_pre)]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    # --- Per-class caps + final K ---
    order = torch.argsort(scores, descending=True)
    kept_idx = []
    counts = Counter()

    for idx in order.tolist():
        lab = int(labels[idx].item())
        name = _label_name(id2label, lab)
        cap = int(per_class_cap.get(name, default_cap))
        if counts[name] < cap:
            kept_idx.append(idx)
            counts[name] += 1
        if final_k and len(kept_idx) >= int(final_k):
            break

    if not kept_idx:
        return boxes[:0], scores[:0], labels[:0]

    kept = torch.tensor(kept_idx, dtype=torch.long, device=boxes.device)
    return boxes[kept], scores[kept], labels[kept]

def _load_gt(graph_path: str, dataset, w: int, h: int):
    """
    Returns:
    gt_boxes: List[List[float]] in xyxy
    gt_labels: List[int]
    gt_areas: List[float] in px^2
    """
    with open(graph_path, "r") as f:
        graph = json.load(f)

    gt_boxes, gt_labels, gt_areas = [], [], []

    for node in graph.get("nodes", []):
        bbox = node.get("bbox")
        if not bbox:
            continue
        clamped = clamp_bbox_xywh(bbox, w, h)
        if clamped is None:
            continue
        x, y, bw, bh = clamped

        raw = node.get("data_class", "") or node.get("category", "")
        l2 = dataset.map_raw_to_l2(str(raw).strip())
        if l2 not in dataset.label2id:
            continue

        gt_boxes.append([x, y, x + bw, y + bh])
        gt_labels.append(dataset.label2id[l2])
        gt_areas.append(float(bw * bh))

    return gt_boxes, gt_labels, gt_areas


def _precision(tp: int, fp: int) -> float:
    d = tp + fp
    return float(tp) / float(d) if d > 0 else 0.0


def _recall(tp: int, fn: int) -> float:
    d = tp + fn
    return float(tp) / float(d) if d > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _coco_size_bucket(area_px: float) -> str:
    # COCO convention (absolute pixel area):
    # small < 32^2, medium < 96^2, large >= 96^2
    if area_px < 32.0 * 32.0:
        return "small"
    if area_px < 96.0 * 96.0:
        return "medium"
    return "large"


def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)
def _rgb_to_hex(c) -> str:
    if isinstance(c, (tuple, list)) and len(c) >= 3:
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        return f"#{r:02x}{g:02x}{b:02x}"
    # fallback (already hex string etc.)
    return str(c)


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _export_overlay_svg(
    out_path: str,
    w: int,
    h: int,
    img_path: str,
    gt_boxes_xyxy: List[List[float]],
    gt_labels: List[int],
    pred_boxes_xyxy: List[List[float]],
    pred_labels: List[int],
    pred_scores: List[float],
    id2label: dict,
    opacity: float = 0.5,
    include_image: bool = True,
):
    """
    Exports an SVG with:
      - optional raster background image
      - GT boxes first (dashed)
      - Pred boxes second (solid) => "on top"
    All rectangles use `opacity` so overlaps are easy to see.
    """
    _safe_mkdir(os.path.dirname(out_path))

    # Use a relative path so the SVG stays portable *within* the run folder.
    rel_img = os.path.relpath(img_path, os.path.dirname(out_path)).replace("\\", "/")

    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    )

    if include_image:
        # Background image (keeps file size small; requires the image file to remain reachable via rel path)
        svg.append(
            f'<image x="0" y="0" width="{w}" height="{h}" '
            f'href="{_xml_escape(rel_img)}" xlink:href="{_xml_escape(rel_img)}" />'
        )

    svg.append("<g id='gt'>")
    for box, lab in zip(gt_boxes_xyxy, gt_labels):
        x0, y0, x1, y1 = [float(v) for v in box]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        if bw <= 0.0 or bh <= 0.0:
            continue

        name = _label_name(id2label, int(lab))
        col = _rgb_to_hex(CLASS_COLORS.get(name, (0, 255, 0)))

        svg.append(
            f"<rect x='{x0:.2f}' y='{y0:.2f}' width='{bw:.2f}' height='{bh:.2f}' "
            f"fill='none' stroke='{col}' stroke-width='3' stroke-dasharray='8,5' "
            f"opacity='{float(opacity):.3f}'>"
            f"<title>{_xml_escape('GT ' + name)}</title>"
            f"</rect>"
        )
    svg.append("</g>")

    # Pred on top
    svg.append("<g id='pred'>")
    for box, lab, sc in zip(pred_boxes_xyxy, pred_labels, pred_scores):
        x0, y0, x1, y1 = [float(v) for v in box]
        bw = max(0.0, x1 - x0)
        bh = max(0.0, y1 - y0)
        if bw <= 0.0 or bh <= 0.0:
            continue

        name = _label_name(id2label, int(lab))
        col = _rgb_to_hex(CLASS_COLORS.get(name, (255, 0, 0)))

        svg.append(
            f"<rect x='{x0:.2f}' y='{y0:.2f}' width='{bw:.2f}' height='{bh:.2f}' "
            f"fill='none' stroke='{col}' stroke-width='3' "
            f"opacity='{float(opacity):.3f}'>"
            f"<title>{_xml_escape(f'PRED {name} {float(sc):.3f}')}</title>"
            f"</rect>"
        )
    svg.append("</g>")

    svg.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


# -----------------------------
# PR / AP computation
# -----------------------------
def _compute_pr_ap_for_class(
    class_id: int,
    gt_by_img: Dict[int, List[torch.Tensor]],
    preds: List[Tuple[int, float, torch.Tensor]],
    iou_thr: float,
):
    """
    VOC-style PR/AP at a single IoU threshold.
    gt_by_img[img_idx] = list of gt boxes (xyxy tensors) for this class
    preds = list of (img_idx, score, pred_box tensor)
    """
    # total GT
    total_gt = sum(len(v) for v in gt_by_img.values())
    if total_gt == 0:
        return {
            "total_gt": 0,
            "total_pred": len(preds),
            "ap": 0.0,
            "best_f1": 0.0,
            "best_score_thr": None,
            "precision": [],
            "recall": [],
            "thresholds": [],
        }

    # per-image "used" flags for GT
    used = {img_idx: np.zeros(len(gts), dtype=np.bool_) for img_idx, gts in gt_by_img.items()}

    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(preds_sorted), dtype=np.float32)
    fp = np.zeros(len(preds_sorted), dtype=np.float32)
    thresholds = np.zeros(len(preds_sorted), dtype=np.float32)

    for i, (img_idx, score, pbox) in enumerate(preds_sorted):
        thresholds[i] = float(score)

        gts = gt_by_img.get(img_idx, [])
        if len(gts) == 0:
            fp[i] = 1.0
            continue

        gt_boxes = torch.stack(gts, dim=0)  # [G,4]
        pred_box = pbox.unsqueeze(0)        # [1,4]
        ious = _pairwise_iou_xyxy(gt_boxes, pred_box).squeeze(1)  # [G]

        best_g = int(torch.argmax(ious).item())
        best_iou = float(ious[best_g].item())

        if best_iou >= float(iou_thr) and not used[img_idx][best_g]:
            tp[i] = 1.0
            used[img_idx][best_g] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    rec = tp_cum / float(total_gt)

    # AP via trapezoid on recall axis
    # (Not COCO AP; just a decent single-threshold AP curve)
    ap = float(np.trapz(prec, rec))

    # Best F1 over all possible cutoffs (i.e., taking top-N predictions)
    f1s = np.array([_f1(float(p), float(r)) for p, r in zip(prec, rec)], dtype=np.float32)
    best_idx = int(np.argmax(f1s)) if f1s.size else 0
    best_f1 = float(f1s[best_idx]) if f1s.size else 0.0
    best_thr = float(thresholds[best_idx]) if thresholds.size else None

    return {
        "total_gt": int(total_gt),
        "total_pred": int(len(preds)),
        "ap": ap,
        "best_f1": best_f1,
        "best_score_thr": best_thr,
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "thresholds": thresholds.tolist(),
    }


def _plot_pr_curves(out_path: str, pr_data: Dict[str, Dict[str, Any]]):
    plt.figure()
    for cls, d in pr_data.items():
        if not d["precision"]:
            continue
        plt.plot(d["recall"], d["precision"], label=f"{cls} (AP={d['ap']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class PR curves (single IoU threshold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_score_hist(out_path: str, cls: str, scores_tp: List[float], scores_fp: List[float], scores_conf: List[float]):
    plt.figure()
    bins = 30
    if scores_tp:
        plt.hist(scores_tp, bins=bins, alpha=0.5, label="TP")
    if scores_fp:
        plt.hist(scores_fp, bins=bins, alpha=0.5, label="FP")
    if scores_conf:
        plt.hist(scores_conf, bins=bins, alpha=0.5, label="CONF")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(f"Score distribution: {cls}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_iou_hist(out_path: str, cls: str, ious_tp: List[float], ious_conf: List[float]):
    plt.figure()
    bins = 30
    if ious_tp:
        plt.hist(ious_tp, bins=bins, alpha=0.5, label="TP IoU")
    if ious_conf:
        plt.hist(ious_conf, bins=bins, alpha=0.5, label="CONF IoU")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.title(f"IoU distribution: {cls}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_confusion_matrix(out_path: str, labels: List[str], mat: np.ndarray):
    plt.figure(figsize=(8, 7))
    plt.imshow(mat, interpolation="nearest")
    plt.title("Confusion (GT rows -> Pred cols) at IoU>=thr (includes MISS col)")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(fraction=0.046, pad=0.04)

    # annotate counts (only if not too huge)
    maxv = mat.max() if mat.size else 0
    thresh = maxv * 0.5 if maxv > 0 else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v == 0:
                continue
            plt.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_size_bucket(out_path: str, buckets: Dict[str, Dict[str, float]]):
    # buckets: bucket -> dict with strict_recall, loc_recall, count
    keys = ["small", "medium", "large"]
    strict = [buckets.get(k, {}).get("strict_recall", 0.0) for k in keys]
    loc = [buckets.get(k, {}).get("loc_recall", 0.0) for k in keys]
    counts = [int(buckets.get(k, {}).get("count", 0)) for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, strict, width, label="Strict (class-aware)")
    plt.bar(x + width / 2, loc, width, label="Localization-only")
    plt.xticks(x, [f"{k}\n(n={c})" for k, c in zip(keys, counts)])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recall")
    plt.title("Recall by GT size bucket")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Visualization drawing
# -----------------------------
def _draw_gt_panel(img, gt_boxes, gt_labels, id2label, font):
    gt_img = img.copy()
    dr = ImageDraw.Draw(gt_img)
    for box, lab in zip(gt_boxes, gt_labels):
        name = _label_name(id2label, int(lab))
        color = CLASS_COLORS.get(name, (0, 255, 0))
        dr.rectangle(box, outline=color, width=3)
        dr.text((box[0] + 2, box[1] + 2), name, fill=color, font=font)
    return gt_img


def _draw_pred_panel(img, pred_boxes, pred_labels, pred_scores, id2label, font):
    pred_img = img.copy()
    dr = ImageDraw.Draw(pred_img)
    for box, lab, score in zip(pred_boxes, pred_labels, pred_scores):
        name = _label_name(id2label, int(lab))
        color = CLASS_COLORS.get(name, (255, 0, 0))
        dr.rectangle(box, outline=color, width=3)
        dr.text((box[0] + 2, box[1] + 2), f"{name} {score:.2f}", fill=color, font=font)
    return pred_img


def _draw_overlay_panel(
    img,
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_to_pred,
    pred_to_gt,
    id2label,
    font,
):
    """
    Overlay encodes:
    - GT missed: red (FN)
    - GT matched correct: green
    - GT matched wrong: purple + "CONF gt->pred"
    - Pred unmatched: orange (FP)
    - Pred matched correct: green "TP"
    - Pred matched wrong: purple "CONF"
    """
    overlay = img.copy()
    dr = ImageDraw.Draw(overlay)

    # GT layer
    for gi, box in enumerate(gt_boxes):
        gt_name = _label_name(id2label, int(gt_labels[gi]))
        if gi not in gt_to_pred:
            dr.rectangle(box, outline=(255, 0, 0), width=3)
            dr.text((box[0] + 2, box[1] + 2), f"FN {gt_name}", fill=(255, 0, 0), font=font)
        else:
            pi = gt_to_pred[gi]
            pred_name = _label_name(id2label, int(pred_labels[pi]))
            if int(pred_labels[pi]) == int(gt_labels[gi]):
                dr.rectangle(box, outline=(0, 180, 0), width=2)
            else:
                dr.rectangle(box, outline=(160, 0, 160), width=3)
                dr.text(
                    (box[0] + 2, box[1] + 2),
                    f"CONF {gt_name}->{pred_name}",
                    fill=(160, 0, 160),
                    font=font,
                )

    # Pred layer
    for pi, box in enumerate(pred_boxes):
        pred_name = _label_name(id2label, int(pred_labels[pi]))
        score = float(pred_scores[pi])
        if pi not in pred_to_gt:
            dr.rectangle(box, outline=(255, 165, 0), width=3)
            dr.text((box[0] + 2, box[1] + 2), f"FP {pred_name} {score:.2f}", fill=(255, 165, 0), font=font)
        else:
            gi = pred_to_gt[pi]
            gt_name = _label_name(id2label, int(gt_labels[gi]))
            if int(pred_labels[pi]) == int(gt_labels[gi]):
                dr.rectangle(box, outline=(0, 200, 0), width=3)
                dr.text((box[0] + 2, box[1] + 2), f"TP {pred_name} {score:.2f}", fill=(0, 200, 0), font=font)
            else:
                dr.rectangle(box, outline=(160, 0, 160), width=3)
                dr.text(
                    (box[0] + 2, box[1] + 2),
                    f"CONF {gt_name}->{pred_name} {score:.2f}",
                    fill=(160, 0, 160),
                    font=font,
                )

    return overlay


def _draw_stats_panel(size, lines, font):
    w, h = size
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    dr = ImageDraw.Draw(img)
    y = 6
    line_h = 12
    for line in lines:
        dr.text((6, y), line, fill=(0, 0, 0), font=font)
        y += line_h
        if y > h - line_h:
            break
    return img


# -----------------------------
# Main analytics containers
# -----------------------------
def _init_counts(level2_classes):
    return {c: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "conf": 0} for c in level2_classes}


@dataclass
class Policy:
    score_thresh: float
    topk_pre: int
    final_k: int
    per_class_cap: dict
    default_cap: int

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=os.path.join(VIS_DIR, "best_test_plus"))
    ap.add_argument("--limit", type=int, default=SUBSET_TEST)
    ap.add_argument("--max_vis", type=int, default=MAX_VIS)

    # shipping policy params
    ap.add_argument("--score_thresh", type=float, default=0.01)
    ap.add_argument("--topk_pre", type=int, default=150)
    ap.add_argument("--final_k", type=int, default=100)
    ap.add_argument("--per_class_cap", type=str, default="")
    ap.add_argument("--default_cap", type=int, default=20)

    # matching / analytics
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # whether to also compute RAW (no policy) analytics
    ap.add_argument("--also_raw", action="store_true", help="Also compute analytics on RAW preds (no final policy).")

    # SVG export
    ap.add_argument("--no_svg", action="store_true", help="Disable SVG export.")
    ap.add_argument("--svg_opacity", type=float, default=0.5, help="Opacity for GT+Pred rectangles in SVG.")
    ap.add_argument("--svg_no_image", action="store_true", help="Do not include raster image as SVG background.")

    args = ap.parse_args()

    _safe_mkdir(args.out_dir)
    plots_dir = os.path.join(args.out_dir, "plots")
    _safe_mkdir(plots_dir)

    svgs_dir = os.path.join(args.out_dir, "svgs")
    _safe_mkdir(svgs_dir)

    device = torch.device(args.device)

    ckpt = os.path.join(CKPT_DIR, "best")
    processor = AutoImageProcessor.from_pretrained(ckpt)
    model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()

    test_dirs = load_split_list(TEST_TXT)
    ds_test = GraphRTDetrDataset(test_dirs, processor, "hierarchy_config.py", augment=False)
    apply_subset(ds_test, args.limit)

    id2label = ds_test.id2label
    level2 = ds_test.level2_classes

    per_class_cap = DEFAULT_PER_CLASS_CAP.copy()
    per_class_cap.update(_parse_caps(args.per_class_cap))

    policy = Policy(
        score_thresh=float(args.score_thresh),
        topk_pre=int(args.topk_pre),
        final_k=int(args.final_k),
        per_class_cap=per_class_cap,
        default_cap=int(args.default_cap),
    )

    font = ImageFont.load_default()
    max_vis = args.max_vis if args.max_vis and args.max_vis > 0 else len(ds_test)

    # Global strict counts (class-aware, confusion penalized as FN+FP)
    global_counts = _init_counts(level2)
    overall = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "conf": 0}

    # Localization-only totals
    loc_overall = {"gt": 0, "matched": 0}
    awr_overall = {"total_area": 0.0, "matched_area": 0.0}

    # Size bucket stats
    bucket = {
        "small": {"count": 0, "strict_tp": 0, "strict_fn": 0, "loc_matched": 0},
        "medium": {"count": 0, "strict_tp": 0, "strict_fn": 0, "loc_matched": 0},
        "large": {"count": 0, "strict_tp": 0, "strict_fn": 0, "loc_matched": 0},
    }

    # Confusion matrix GT->Pred (includes MISS as last column)
    cls_names = [c for c in level2]
    name_to_idx = {c: i for i, c in enumerate(cls_names)}
    miss_col = len(cls_names)
    conf_mat = np.zeros((len(cls_names), len(cls_names) + 1), dtype=np.int64)  # last col = MISS

    # Score / IoU distributions per class
    score_tp = {c: [] for c in level2}
    score_fp = {c: [] for c in level2}
    score_conf = {c: [] for c in level2}
    iou_tp = {c: [] for c in level2}
    iou_conf = {c: [] for c in level2}

    # Store per-image preds/gt for PR curves (policy-applied)
    gt_by_img_per_class: Dict[str, Dict[int, List[torch.Tensor]]] = {c: defaultdict(list) for c in level2}
    preds_per_class: Dict[str, List[Tuple[int, float, torch.Tensor]]] = {c: [] for c in level2}

    # Optionally also store raw per-class predictions for PR (to compare)
    raw_gt_by_img_per_class = {c: defaultdict(list) for c in level2}
    raw_preds_per_class = {c: [] for c in level2}

    # Per-image summaries + worst-case ranking helpers
    image_summaries = []
    worst_by_fn = []
    worst_by_fp = []
    worst_by_conf = []

    # -----------------------------------------
    # Main loop over dataset (policy analytics)
    # -----------------------------------------
    for idx in tqdm(range(len(ds_test)), desc="Best checkpoint: test+analytics"):
        img_path, graph_path, _ = ds_test.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        sample_id = os.path.basename(os.path.dirname(img_path))

        gt_boxes_l, gt_labels_l, gt_areas_l = _load_gt(graph_path, ds_test, w, h)
        gt_total = len(gt_boxes_l)

        # stash GT for PR
        for b, lab in zip(gt_boxes_l, gt_labels_l):
            cls = _label_name(id2label, int(lab))
            gt_by_img_per_class[cls][idx].append(torch.tensor(b, dtype=torch.float32))
            if args.also_raw:
                raw_gt_by_img_per_class[cls][idx].append(torch.tensor(b, dtype=torch.float32))

        # forward
        enc = ds_test.image_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)

        post = ds_test.image_processor.post_process_object_detection(
            out,
            threshold=0.0,
            target_sizes=torch.tensor([[h, w]], device=device),
        )[0]

        # raw tensors (cpu)
        boxes = post["boxes"].detach().cpu()
        scores = post["scores"].detach().cpu()
        labels = post["labels"].detach().cpu()

        boxes, scores, labels = _clip_and_filter_xyxy(boxes, scores, labels, w, h)

        # keep a copy for RAW analytics/PR if requested
        raw_boxes, raw_scores, raw_labels = boxes, scores, labels

        # apply shipping policy
        boxes, scores, labels = _apply_final_policy(
            boxes,
            scores,
            labels,
            id2label=id2label,
            score_thresh=policy.score_thresh,
            topk_pre=policy.topk_pre,
            final_k=policy.final_k,
            per_class_cap=policy.per_class_cap,
            default_cap=policy.default_cap,
        )

        pred_total = int(boxes.shape[0])

        # store preds for PR
        for b, s, lab in zip(boxes, scores, labels):
            cls = _label_name(id2label, int(lab))
            preds_per_class[cls].append((idx, float(s.item()), b.float().clone()))

        if args.also_raw:
            for b, s, lab in zip(raw_boxes, raw_scores, raw_labels):
                cls = _label_name(id2label, int(lab))
                raw_preds_per_class[cls].append((idx, float(s.item()), b.float().clone()))

        # Matching (one-to-one by IoU, class-agnostic assignment)
        gt_boxes_t = torch.tensor(gt_boxes_l, dtype=torch.float32) if gt_boxes_l else torch.zeros((0, 4))
        gt_labels_t = torch.tensor(gt_labels_l, dtype=torch.long) if gt_labels_l else torch.zeros((0,), dtype=torch.long)

        if gt_boxes_t.numel() == 0 and boxes.numel() == 0:
            # nothing in this image
            svg_path = ""
            if (not args.no_svg) and (idx < max_vis):
                svg_path = os.path.join(svgs_dir, f"{idx:05d}_{sample_id}.svg")
                _export_overlay_svg(
                    out_path=svg_path,
                    w=w,
                    h=h,
                    img_path=img_path,
                    gt_boxes_xyxy=[],
                    gt_labels=[],
                    pred_boxes_xyxy=[],
                    pred_labels=[],
                    pred_scores=[],
                    id2label=id2label,
                    opacity=float(args.svg_opacity),
                    include_image=(not args.svg_no_image),
                )

            image_summaries.append({
                "index": idx,
                "id": sample_id,
                "path": img_path,
                "vis_path": "",
                "svg_path": svg_path,
                "gt": 0,
                "pred": pred_total,
                "tp": 0,
                "fp": pred_total,
                "fn": 0,
                "conf": 0,
                "loc_recall": 0.0,
            })
            continue

        gt_to_pred, pred_to_gt, ious = _one_to_one_match_by_iou(gt_boxes_t, boxes, args.iou_thresh)

        # Strict accounting
        tp_pred = set()
        conf_pred = set()
        matched_gt = set(gt_to_pred.keys())

        conf_count = 0
        tp_count = 0

        # Localization-only
        loc_matched = len(matched_gt)
        loc_overall["gt"] += gt_total
        loc_overall["matched"] += loc_matched

        # area-weighted localization
        awr_overall["total_area"] += float(sum(gt_areas_l))
        for gi in matched_gt:
            awr_overall["matched_area"] += float(gt_areas_l[gi])

        # GT-level: TP vs FN (FN includes MISS + CONF)
        fn_count = 0
        for gi in range(gt_total):
            gt_lab = int(gt_labels_t[gi].item())
            gt_name = _label_name(id2label, gt_lab)

            # size bucket
            bkt = _coco_size_bucket(gt_areas_l[gi])
            bucket[bkt]["count"] += 1

            if gi not in gt_to_pred:
                fn_count += 1
                global_counts[gt_name]["gt"] += 1
                global_counts[gt_name]["fn"] += 1
                conf_mat[name_to_idx[gt_name], miss_col] += 1
                bucket[bkt]["strict_fn"] += 1
                continue

            pi = gt_to_pred[gi]
            pred_lab = int(labels[pi].item())
            pred_name = _label_name(id2label, pred_lab)
            iou_val = float(ious[gi, pi].item())

            global_counts[gt_name]["gt"] += 1

            if pred_lab == gt_lab:
                tp_count += 1
                tp_pred.add(pi)
                global_counts[gt_name]["tp"] += 1
                iou_tp[gt_name].append(iou_val)
                bucket[bkt]["strict_tp"] += 1
                bucket[bkt]["loc_matched"] += 1
                conf_mat[name_to_idx[gt_name], name_to_idx[pred_name]] += 1
            else:
                # confusion: counts as FN for GT class, FP for Pred class
                conf_count += 1
                conf_pred.add(pi)
                global_counts[gt_name]["fn"] += 1
                global_counts[gt_name]["conf"] += 1
                iou_conf[gt_name].append(iou_val)
                bucket[bkt]["strict_fn"] += 1
                bucket[bkt]["loc_matched"] += 1
                conf_mat[name_to_idx[gt_name], name_to_idx[pred_name]] += 1

        # Pred-level: FP includes unmatched + confusion preds
        fp_count = 0
        for pi in range(pred_total):
            pred_lab = int(labels[pi].item())
            pred_name = _label_name(id2label, pred_lab)
            global_counts[pred_name]["pred"] += 1

            sc = float(scores[pi].item())
            if pi in tp_pred:
                score_tp[pred_name].append(sc)
            elif pi in conf_pred:
                # This pred hit some GT but as wrong class -> confusion
                score_conf[pred_name].append(sc)
                global_counts[pred_name]["fp"] += 1
                fp_count += 1
            else:
                # unmatched -> FP
                score_fp[pred_name].append(sc)
                global_counts[pred_name]["fp"] += 1
                fp_count += 1

        # FN count already computed as (gt_total - tp_count) with confusion included,
        # but we tracked explicitly per GT to keep bucket stats consistent.
        fn_count = (gt_total - tp_count)

        # Overall tallies
        overall["gt"] += gt_total
        overall["pred"] += pred_total
        overall["tp"] += tp_count
        overall["fp"] += fp_count
        overall["fn"] += fn_count
        overall["conf"] += conf_count

        loc_recall = float(loc_matched) / float(gt_total) if gt_total > 0 else 0.0

        vis_path = os.path.join(args.out_dir, f"{idx:05d}_{sample_id}.png")

        svg_path = ""
        if (not args.no_svg) and (idx < max_vis):
            svg_path = os.path.join(svgs_dir, f"{idx:05d}_{sample_id}.svg")
            _export_overlay_svg(
                out_path=svg_path,
                w=w,
                h=h,
                img_path=img_path,
                gt_boxes_xyxy=gt_boxes_l,
                gt_labels=gt_labels_l,
                pred_boxes_xyxy=boxes.tolist(),
                pred_labels=labels.tolist(),
                pred_scores=scores.tolist(),
                id2label=id2label,
                opacity=float(args.svg_opacity),
                include_image=(not args.svg_no_image),
            )

        # Visuals for first max_vis images
        if idx < max_vis:
            gt_img = _draw_gt_panel(img, gt_boxes_l, gt_labels_l, id2label, font)
            pred_img = _draw_pred_panel(img, boxes.tolist(), labels.tolist(), scores.tolist(), id2label, font)
            overlay_img = _draw_overlay_panel(
                img,
                gt_boxes_l,
                gt_labels_l,
                boxes.tolist(),
                labels.tolist(),
                scores.tolist(),
                gt_to_pred,
                pred_to_gt,
                id2label,
                font,
            )

            # per-image per-class counts
            per_img = _init_counts(level2)
            for lab in gt_labels_l:
                per_img[_label_name(id2label, int(lab))]["gt"] += 1
            for lab in labels.tolist():
                per_img[_label_name(id2label, int(lab))]["pred"] += 1

            # derive per-image strict tp/fn/conf based on gt_to_pred assignments
            for gi in range(gt_total):
                gt_lab = int(gt_labels_t[gi].item())
                gt_name = _label_name(id2label, gt_lab)
                if gi not in gt_to_pred:
                    per_img[gt_name]["fn"] += 1
                else:
                    pi = gt_to_pred[gi]
                    pred_lab = int(labels[pi].item())
                    if pred_lab == gt_lab:
                        per_img[gt_name]["tp"] += 1
                    else:
                        per_img[gt_name]["fn"] += 1
                        per_img[gt_name]["conf"] += 1

            # pred-side FP: unmatched + confusion preds
            for pi in range(pred_total):
                pred_name = _label_name(id2label, int(labels[pi].item()))
                if pi not in tp_pred:
                    per_img[pred_name]["fp"] += 1

            lines = [
                f"Image: {sample_id}",
                f"GT {gt_total}  Pred {pred_total}",
                f"TP {tp_count}  FP {fp_count}  FN {fn_count}  CONF {conf_count}",
                f"IoU >= {args.iou_thresh:.2f}",
                f"Score >= {policy.score_thresh:.2f} | topk_pre={policy.topk_pre} final_k={policy.final_k}",
                f"LocRecall={loc_recall:.3f}",
                "",
                "Per-class (strict):",
            ]
            for cls in level2:
                s = per_img[cls]
                p = _precision(s["tp"], s["fp"])
                r = _recall(s["tp"], s["fn"])
                lines.append(f"{cls}: P {p:.2f} R {r:.2f} TP {s['tp']} FP {s['fp']} FN {s['fn']} CONF {s['conf']}")

            stats_img = _draw_stats_panel((w, h), lines, font)

            canvas = Image.new("RGB", (w * 2, h * 2), color=(255, 255, 255))
            canvas.paste(gt_img, (0, 0))
            canvas.paste(pred_img, (w, 0))
            canvas.paste(overlay_img, (0, h))
            canvas.paste(stats_img, (w, h))
            canvas.save(vis_path)

        # store summary
        s = {
            "index": idx,
            "id": sample_id,
            "path": img_path,
            "vis_path": vis_path if idx < max_vis else "",
            "svg_path": svg_path if idx < max_vis else "",
            "gt": gt_total,
            "pred": pred_total,
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "conf": conf_count,
            "strict_precision": _precision(tp_count, fp_count),
            "strict_recall": _recall(tp_count, fn_count),
            "strict_f1": _f1(_precision(tp_count, fp_count), _recall(tp_count, fn_count)),
            "loc_recall": loc_recall,
        }
        image_summaries.append(s)

        worst_by_fn.append((fn_count, idx, sample_id))
        worst_by_fp.append((fp_count, idx, sample_id))
        worst_by_conf.append((conf_count, idx, sample_id))

    # -----------------------------------------
    # Aggregate summaries
    # -----------------------------------------
    per_class_summary = {}
    for cls in level2:
        s = global_counts[cls]
        per_class_summary[cls] = {
            **s,
            "precision": _precision(s["tp"], s["fp"]),
            "recall": _recall(s["tp"], s["fn"]),
            "f1": _f1(_precision(s["tp"], s["fp"]), _recall(s["tp"], s["fn"])),
        }

    overall_prec = _precision(overall["tp"], overall["fp"])
    overall_rec = _recall(overall["tp"], overall["fn"])
    overall_f1 = _f1(overall_prec, overall_rec)

    loc_recall_total = float(loc_overall["matched"]) / float(loc_overall["gt"]) if loc_overall["gt"] > 0 else 0.0
    awr = float(awr_overall["matched_area"]) / float(max(awr_overall["total_area"], 1e-6))

    # size bucket recall
    bucket_out = {}
    for k in ["small", "medium", "large"]:
        c = int(bucket[k]["count"])
        strict_tp = int(bucket[k]["strict_tp"])
        strict_fn = int(bucket[k]["strict_fn"])
        loc_m = int(bucket[k]["loc_matched"])
        strict_rec = float(strict_tp) / float(strict_tp + strict_fn) if (strict_tp + strict_fn) > 0 else 0.0
        loc_rec = float(loc_m) / float(c) if c > 0 else 0.0
        bucket_out[k] = {
            "count": c,
            "strict_recall": strict_rec,
            "loc_recall": loc_rec,
        }

    # -----------------------------------------
    # PR curves + AP (policy-applied)
    # -----------------------------------------
    pr_data = {}
    for cls in level2:
        cls_id = ds_test.label2id[cls]
        pr = _compute_pr_ap_for_class(
            class_id=cls_id,
            gt_by_img=gt_by_img_per_class[cls],
            preds=preds_per_class[cls],
            iou_thr=float(args.iou_thresh),
        )
        pr_data[cls] = pr

    # -----------------------------------------
    # Save plots
    # -----------------------------------------
    _plot_pr_curves(os.path.join(plots_dir, "pr_curves.png"), pr_data)
    _plot_size_bucket(os.path.join(plots_dir, "size_bucket_recall.png"), bucket_out)

    # per-class score/iou hists
    for cls in level2:
        _plot_score_hist(
            os.path.join(plots_dir, f"scores_{cls}.png"),
            cls,
            score_tp[cls],
            score_fp[cls],
            score_conf[cls],
        )
        _plot_iou_hist(
            os.path.join(plots_dir, f"ious_{cls}.png"),
            cls,
            iou_tp[cls],
            iou_conf[cls],
        )

    # confusion matrix plot (add MISS label)
    conf_labels = cls_names + ["MISS"]
    _plot_confusion_matrix(os.path.join(plots_dir, "confusion_matrix.png"), conf_labels, conf_mat)

    # -----------------------------------------
    # Worst-case lists
    # -----------------------------------------
    worst_by_fn.sort(reverse=True)
    worst_by_fp.sort(reverse=True)
    worst_by_conf.sort(reverse=True)

    worst = {
        "most_fn": [{"fn": a, "index": i, "id": sid} for a, i, sid in worst_by_fn[:50]],
        "most_fp": [{"fp": a, "index": i, "id": sid} for a, i, sid in worst_by_fp[:50]],
        "most_conf": [{"conf": a, "index": i, "id": sid} for a, i, sid in worst_by_conf[:50]],
    }

    # -----------------------------------------
    # Final summary dumps
    # -----------------------------------------
    summary = {
        "config": {
            "iou_thresh": float(args.iou_thresh),
            "policy": {
                "score_thresh": policy.score_thresh,
                "topk_pre": policy.topk_pre,
                "final_k": policy.final_k,
                "per_class_cap": policy.per_class_cap,
                "default_cap": policy.default_cap,
            },
            "limit": int(args.limit),
            "max_vis": int(max_vis),
            "also_raw": bool(args.also_raw),
            "svg": {
                "enabled": (not args.no_svg),
                "opacity": float(args.svg_opacity),
                "include_image": (not args.svg_no_image),
                "svgs_dir": svgs_dir,
            },
        },
        "overall_strict": {
            **overall,
            "precision": overall_prec,
            "recall": overall_rec,
            "f1": overall_f1,
        },
        "overall_localization": {
            **loc_overall,
            "loc_recall": loc_recall_total,
            "area_weighted_loc_recall": awr,
        },
        "per_class_strict": per_class_summary,
        "pr_ap": {
            cls: {
                "ap": pr_data[cls]["ap"],
                "best_f1": pr_data[cls]["best_f1"],
                "best_score_thr": pr_data[cls]["best_score_thr"],
            }
            for cls in level2
        },
        "size_bucket_recall": bucket_out,
        "confusion_matrix": {
            "labels": conf_labels,
            "matrix": conf_mat.tolist(),
        },
        "worst_cases": worst,
        "images": image_summaries,
        "plots_dir": plots_dir,
        "svgs_dir": svgs_dir,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    summary_txt = os.path.join(args.out_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write("=== Overall (Strict / class-aware) ===\n")
        f.write(
            f"GT={overall['gt']} Pred={overall['pred']} TP={overall['tp']} FP={overall['fp']} "
            f"FN={overall['fn']} CONF={overall['conf']}\n"
        )
        f.write(f"Precision={overall_prec:.4f} Recall={overall_rec:.4f} F1={overall_f1:.4f}\n\n")

        f.write("=== Overall (Localization-only) ===\n")
        f.write(f"GT={loc_overall['gt']} Matched={loc_overall['matched']} LocRecall={loc_recall_total:.4f}\n")
        f.write(f"AreaWeightedLocRecall={awr:.4f}\n\n")

        f.write("=== Per-class (Strict) ===\n")
        for cls in level2:
            s = per_class_summary[cls]
            f.write(
                f"{cls}: P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} "
                f"TP={s['tp']} FP={s['fp']} FN={s['fn']} CONF={s['conf']} GT={s['gt']} Pred={s['pred']}\n"
            )

        f.write("\n=== PR/AP (single IoU threshold) ===\n")
        for cls in level2:
            d = pr_data[cls]
            f.write(f"{cls}: AP={d['ap']:.4f} BestF1={d['best_f1']:.4f} BestThr={d['best_score_thr']}\n")

        f.write("\n=== Size bucket recall ===\n")
        for k in ["small", "medium", "large"]:
            d = bucket_out[k]
            f.write(f"{k}: n={d['count']} strict_recall={d['strict_recall']:.4f} loc_recall={d['loc_recall']:.4f}\n")

        f.write("\n=== Worst cases ===\n")
        f.write("Most FN:\n")
        for it in worst["most_fn"][:10]:
            f.write(f"  idx={it['index']} id={it['id']} fn={it['fn']}\n")
        f.write("Most FP:\n")
        for it in worst["most_fp"][:10]:
            f.write(f"  idx={it['index']} id={it['id']} fp={it['fp']}\n")
        f.write("Most CONF:\n")
        for it in worst["most_conf"][:10]:
            f.write(f"  idx={it['index']} id={it['id']} conf={it['conf']}\n")

    print(f"Saved visuals to: {args.out_dir}")
    print(f"Saved SVGs to:    {svgs_dir}")
    print(f"Saved plots to:   {plots_dir}")
    print(f"Saved summary:    {summary_path}")


if __name__ == "__main__":
    main()
