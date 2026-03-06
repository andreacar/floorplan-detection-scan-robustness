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
from PIL import Image
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


def _features_from_backbone(model, processor, image: Image.Image, device: torch.device):
    enc = processor(images=image, return_tensors="pt").to(device)
    pixel_values = enc["pixel_values"]
    pixel_mask = enc.get("pixel_mask")
    if pixel_mask is None:
        pixel_mask = torch.ones((pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]), device=device,
                                dtype=torch.bool)
    else:
        pixel_mask = pixel_mask.to(device).bool()

    with torch.no_grad():
        features = model.model.backbone(pixel_values, pixel_mask)
    return features


def _compute_patch_distances(feats_clean, feats_scan) -> List[np.ndarray]:
    maps = []
    for (feat_clean, _), (feat_scan, _) in zip(feats_clean, feats_scan):
        diff = torch.norm(feat_clean - feat_scan, dim=1).squeeze(0)
        maps.append(diff.cpu().numpy())
    return maps


def _object_level_shifts(
    boxes: List[List[float]],
    labels: List[int],
    dist_maps: List[np.ndarray],
    strides: List[int],
    id2label: Dict[int, str],
) -> List[Dict]:
    out = []
    for box, lbl in zip(boxes, labels):
        level_scores = {}
        for lvl, dist_map in enumerate(dist_maps):
            stride = strides[min(lvl, len(strides) - 1)]
            h, w = dist_map.shape
            x1, y1, x2, y2 = box
            xs = int(np.clip(np.floor(x1 / stride), 0, w - 1))
            ys = int(np.clip(np.floor(y1 / stride), 0, h - 1))
            xe = int(np.clip(np.ceil(x2 / stride), 0, w))
            ye = int(np.clip(np.ceil(y2 / stride), 0, h))
            patch = dist_map[ys:ye, xs:xe]
            level_scores[f"level_{lvl}"] = float(patch.mean()) if patch.size else 0.0
        out.append(
            {
                "label": id2label.get(lbl, str(lbl)),
                "box": [float(v) for v in box],
                "level_means": level_scores,
            }
        )
    return out


def _save_heatmaps(dist_maps: List[np.ndarray], layout_id: str, out_dir: str):
    heat_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)
    for lvl, dist_map in enumerate(dist_maps):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        im = ax.imshow(dist_map, cmap="inferno")
        ax.axis("off")
        ax.set_title(f"{layout_id} L{lvl}")
        fig.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(heat_dir, f"{layout_id}_level{lvl}.png"), dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Patch-level representation shift localization.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="paper_experiments/out/shift_maps")
    parser.add_argument("--image-a", default="model_baked.png")
    parser.add_argument("--image-b", default="F1_scaled.png")
    parser.add_argument("--test-txt", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--severity-steps", type=int, default=3)
    args = parser.parse_args()

    safe_makedirs(args.out_dir)
    device = torch.device(args.device)
    level2, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(args.ckpt, device)
    test_dirs = load_test_dirs(args.test_txt)
    if args.limit:
        test_dirs = test_dirs[: args.limit]

    config = model.model.config
    strides = list(config.feat_strides)

    records = []
    for folder in tqdm(test_dirs, desc="Layout pairs"):
        img_a_path = os.path.join(folder, args.image_a)
        img_b_path = os.path.join(folder, args.image_b)
        graph_path = os.path.join(folder, "graph.json")
        if not (os.path.exists(img_a_path) and os.path.exists(img_b_path) and os.path.exists(graph_path)):
            continue
        img_a = load_image(img_a_path)
        img_b = load_image(img_b_path)

        gt_boxes_a, gt_labels_a, _ = load_gt_from_graph(graph_path, img_a.size[0], img_a.size[1], map_raw_to_l2, label2id)
        gt_boxes_b, gt_labels_b, _ = load_gt_from_graph(graph_path, img_b.size[0], img_b.size[1], map_raw_to_l2, label2id)

        stats = {}
        feats_a = _features_from_backbone(model, processor, img_a, device)
        feats_b = _features_from_backbone(model, processor, img_b, device)
        dist_maps = _compute_patch_distances(feats_a, feats_b)
        _save_heatmaps(dist_maps, os.path.basename(folder.rstrip("/")), args.out_dir)

        boxes_a, scores_a, labels_a = infer_predictions(
            model,
            processor,
            img_a,
            device,
            id2label,
            score_thresh=0.0,
            topk_pre=0,
            final_k=0,
            per_class_cap=DEFAULT_PER_CLASS_CAP.copy(),
            use_per_class_thresh=False,
        )
        boxes_b, scores_b, labels_b = infer_predictions(
            model,
            processor,
            img_b,
            device,
            id2label,
            score_thresh=0.0,
            topk_pre=0,
            final_k=0,
            per_class_cap=DEFAULT_PER_CLASS_CAP.copy(),
            use_per_class_thresh=False,
        )

        recall_a = float(sum(i >= 0.85 for i in match_greedy_by_class(gt_boxes_a, gt_labels_a, boxes_a.tolist(), scores_a.tolist(), labels_a.tolist(), iou_thr=0.5))) / max(
            len(gt_labels_a), 1
        )
        recall_b = float(sum(i >= 0.85 for i in match_greedy_by_class(gt_boxes_b, gt_labels_b, boxes_b.tolist(), scores_b.tolist(), labels_b.tolist(), iou_thr=0.5))) / max(
            len(gt_labels_b), 1
        )

        stats["layout_id"] = os.path.basename(folder.rstrip("/"))
        stats["global_dist"] = float(np.linalg.norm(extract_embedding(model, processor, img_a, device) - extract_embedding(model, processor, img_b, device)))
        stats["recall85_clean"] = recall_a
        stats["recall85_scanned"] = recall_b
        stats["level_stats"] = [{"level": lvl, "mean": float(np.mean(dist_map)), "max": float(np.max(dist_map))} for lvl, dist_map in enumerate(dist_maps)]
        stats["object_distances"] = _object_level_shifts(gt_boxes_b, gt_labels_b, dist_maps, strides, id2label)
        records.append(stats)

    summary_path = os.path.join(args.out_dir, "shift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} layouts + heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
