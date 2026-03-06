#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from torchvision.ops import box_iou
from transformers import AutoImageProcessor, RTDetrForObjectDetection
from transformers.image_transforms import center_to_corners_format

from utils.paths import load_split_list
from data.dataset import GraphRTDetrDataset, apply_subset
from config import *


# ----------------------------
# Temperature scaler
# ----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T = exp(log_T)

    def forward(self, logits):
        return logits / torch.exp(self.log_T)


# ----------------------------
# Box conversion helpers
# ----------------------------
def _to_xyxy_norm_from_cxcywh(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    # boxes: [N,4] in cx,cy,w,h normalized
    xyxy = center_to_corners_format(boxes_cxcywh)
    return xyxy.clamp(0.0, 1.0)


def _to_xyxy_norm_from_xywh(boxes_xywh: torch.Tensor) -> torch.Tensor:
    # boxes: [N,4] in x,y,w,h normalized (top-left)
    x, y, w, h = boxes_xywh.unbind(-1)
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def _guess_and_convert_gt_to_xyxy_norm(gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Robustly convert GT boxes to xyxy normalized.
    Handles likely formats:
      - cxcywh normalized (most common in HF DETR training)
      - xyxy normalized
      - xywh normalized (top-left)
    """
    if gt_boxes.numel() == 0:
        return gt_boxes

    b = gt_boxes.clone()

    # Heuristic 1: if most boxes have x2>x1 and y2>y1 => likely xyxy
    frac_xyxy = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).float().mean().item()

    # If it looks like xyxy, keep it
    if frac_xyxy > 0.95:
        return b.clamp(0.0, 1.0)

    # Heuristic 2: decide between cxcywh vs xywh
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    left_neg = (cx - w / 2 < -0.01).float().mean().item()
    right_over = (cx + w / 2 > 1.01).float().mean().item()
    top_neg = (cy - h / 2 < -0.01).float().mean().item()
    bot_over = (cy + h / 2 > 1.01).float().mean().item()
    cxcywh_score = left_neg + right_over + top_neg + bot_over

    if cxcywh_score > 0.02:
        return _to_xyxy_norm_from_cxcywh(b)
    else:
        # if it didn't look like xyxy and doesn't violate like cxcywh, assume xywh
        return _to_xyxy_norm_from_xywh(b)


def xyxy_norm_to_abs(xyxy_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
    out = xyxy_norm.clone()
    out[:, [0, 2]] *= W
    out[:, [1, 3]] *= H
    return out


# ----------------------------
# Query boxes + scores with indices (no processor needed)
# ----------------------------
@torch.no_grad()
def get_queries_xyxy_abs_scores_labels(outputs, H: int, W: int):
    """
    Rebuild what post_process_object_detection does, but keep query indices aligned.
    - outputs.pred_boxes: [B,Q,4] in cxcywh normalized
    - outputs.logits: [B,Q,C] where last class is typically "no-object"
    """
    logits = outputs.logits[0]          # [Q,C]
    pred_boxes = outputs.pred_boxes[0]  # [Q,4] cxcywh norm

    probs = logits.softmax(-1)

    # DETR-style: exclude the last "no_object" class from argmax
    if probs.shape[-1] >= 2:
        scores, labels = probs[:, :-1].max(dim=-1)
    else:
        scores, labels = probs.max(dim=-1)

    # convert boxes to xyxy abs
    xyxy_norm = _to_xyxy_norm_from_cxcywh(pred_boxes)
    boxes_abs = xyxy_norm_to_abs(xyxy_norm, H, W)

    # query indices are 0..Q-1
    q_idx = torch.arange(logits.shape[0], device=logits.device)
    return boxes_abs, scores, labels, logits, q_idx


# ----------------------------
# Greedy 1-to-1 matching
# ----------------------------
def greedy_match(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh=0.3):
    """
    pred_boxes: [P,4] xyxy abs
    gt_boxes:   [G,4] xyxy abs
    returns list of (p_idx, g_idx) matched 1-to-1 by IoU descending
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []

    ious = box_iou(pred_boxes, gt_boxes)  # [P,G]
    p_idx, g_idx = torch.where(ious >= iou_thresh)
    if p_idx.numel() == 0:
        return []

    vals = ious[p_idx, g_idx]
    order = torch.argsort(vals, descending=True)

    used_p = set()
    used_g = set()
    matches = []
    for k in order.tolist():
        p = int(p_idx[k])
        g = int(g_idx[k])
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        matches.append((p, g))
    return matches


# ----------------------------
# Analysis
# ----------------------------
@torch.no_grad()
def analyze_scores(model, processor, ds_val, device, T=None):
    score_stats = defaultdict(list)
    id2label = model.config.id2label

    for sample in tqdm(ds_val):

        b = sample["labels"]["boxes"]
        H, W = sample["labels"]["orig_size"].tolist()
        print("orig_size H,W:", H, W)
        print("boxes shape:", b.shape)
        print("boxes min/max:", float(b.min()), float(b.max()))
        print("first 10 boxes:", b[:10])
        break

        pixel = sample["pixel_values"].unsqueeze(0).to(device)
        pixel_mask = torch.ones(
            (1, pixel.shape[-2], pixel.shape[-1]),
            dtype=torch.long,
            device=device
        )

        outputs = model(pixel_values=pixel, pixel_mask=pixel_mask)
        if T is not None:
            outputs.logits = outputs.logits / T

        H, W = sample["labels"]["orig_size"].tolist()

        processed = processor.post_process_object_detection(
            outputs,
            threshold=0.0,
            target_sizes=torch.tensor([[H, W]], device=device)
        )[0]

        for lbl, score in zip(processed["labels"].cpu(), processed["scores"].cpu()):
            score_stats[int(lbl)].append(float(score))

    for k in sorted(score_stats):
        s = score_stats[k]
        print(
            f"{id2label[k]:>8s} | "
            f"mean={np.mean(s):.3f} "
            f"p90={np.percentile(s,90):.3f} "
            f"max={np.max(s):.3f}"
        )

    return score_stats


# ----------------------------
# Collect matched detection logits + targets
# ----------------------------
@torch.no_grad()
def collect_logits_and_targets_matched(
    model,
    ds_val,
    device,
    iou_thresh=0.3,
    min_score=0.0,
    top_k=300
):
    all_logits = []
    all_targets = []

    print("Collecting matched pairs for temperature scaling…")

    for sample in tqdm(ds_val):
        pixel = sample["pixel_values"].unsqueeze(0).to(device)
        pixel_mask = torch.ones(
            (1, pixel.shape[-2], pixel.shape[-1]),
            dtype=torch.long,
            device=device
        )

        outputs = model(pixel_values=pixel, pixel_mask=pixel_mask)

        H, W = sample["labels"]["orig_size"].tolist()

        # per-query boxes/scores/labels/logits with query indices
        pred_boxes_abs, scores, labels, logits, q_idx = get_queries_xyxy_abs_scores_labels(outputs, H, W)

        # ---- NEW: keep only top-K queries to avoid matching against junk ----
        K = min(top_k, scores.shape[0])
        scores_k, idx_k = torch.topk(scores, k=K)
        pred_boxes_abs = pred_boxes_abs[idx_k]
        logits = logits[idx_k]
        scores = scores_k

        # optional score floor
        if min_score > 0.0:
            keep = scores >= min_score
            pred_boxes_abs = pred_boxes_abs[keep]
            logits = logits[keep]
            scores = scores[keep]

        # GT boxes: convert robustly to xyxy abs
        gt_boxes_norm = sample["labels"]["boxes"].to(device)
        gt_boxes_xyxy_norm = _guess_and_convert_gt_to_xyxy_norm(gt_boxes_norm)
        gt_boxes_abs = xyxy_norm_to_abs(gt_boxes_xyxy_norm, H, W)
        gt_labels = sample["labels"]["class_labels"].to(device)

        matches = greedy_match(pred_boxes_abs, gt_boxes_abs, iou_thresh=iou_thresh)
        if not matches:
            continue

        p_inds = torch.tensor([m[0] for m in matches], device=device, dtype=torch.long)
        g_inds = torch.tensor([m[1] for m in matches], device=device, dtype=torch.long)

        all_logits.append(logits[p_inds])
        all_targets.append(gt_labels[g_inds])

    if not all_logits:
        raise RuntimeError(
            "No matched detections found. GT box format may be unexpected.\n"
            "Try lowering iou_thresh further (e.g. 0.1) or inspect sample['labels']['boxes']."
        )

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


# ----------------------------
# Main
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading BEST model + processor…")
    ckpt = os.path.join(CKPT_DIR, "best")
    processor = AutoImageProcessor.from_pretrained(ckpt)
    model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device)
    model.eval()

    print("Loading validation dataset…")
    val_dirs = load_split_list(VAL_TXT)
    ds_val = GraphRTDetrDataset(val_dirs, processor, "hierarchy_config.py", augment=False)
    apply_subset(ds_val, SUBSET_VAL)

    print("\n=== ANALYSIS BEFORE TEMPERATURE SCALING ===")
    analyze_scores(model, processor, ds_val, device, T=None)

    logits, targets = collect_logits_and_targets_matched(
        model=model,
        ds_val=ds_val,
        device=device,
        iou_thresh=0.3,   # <-- relaxed for calibration stability
        min_score=0.0,
        top_k=300         # <-- key change
    )

    print(f"Collected {logits.shape[0]} matched detections for calibration")

    # Fit temperature
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = torch.exp(scaler.log_T).item()
    print(f"\n>>> Optimal temperature learned: T = {T:.3f}")

    print("\n=== ANALYSIS AFTER TEMPERATURE SCALING ===")
    analyze_scores(model, processor, ds_val, device, T=T)

    print("\nDone.")


if __name__ == "__main__":
    main()
