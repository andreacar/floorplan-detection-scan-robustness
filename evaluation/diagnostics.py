from collections import defaultdict
import json
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from utils.geometry import clamp_bbox_xywh, compute_iou
from models.detector_utils import detector_predict_post


def _clip_and_filter_xyxy(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, w: int, h: int):
    """
    Clip xyxy boxes to image bounds and drop degenerate boxes.
    Returns (boxes, scores, labels) filtered.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    b = boxes.clone()
    b[:, 0] = b[:, 0].clamp(0, w)
    b[:, 2] = b[:, 2].clamp(0, w)
    b[:, 1] = b[:, 1].clamp(0, h)
    b[:, 3] = b[:, 3].clamp(0, h)

    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep], scores[keep], labels[keep]


def compute_classwise_recall_and_iou(
    model,
    dataset,
    device,
    iou_thresh: float = 0.5,
    score_thresh: float = 0.0,
    topk: int | None = 100,
):
    """
    Class-wise recall@IoU and IoU stats.
    - GT boxes are clamped to image bounds (drop fully-outside).
    - Pred boxes are clipped to image bounds (drop degenerate).
    Returns a summary dict with config + per-class metrics.
    """
    def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: [N,4], b: [M,4] in xyxy (pixel coords)
        if a.numel() == 0 or b.numel() == 0:
            return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)

        area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

        lt = torch.max(a[:, None, :2], b[:, :2])
        rb = torch.min(a[:, None, 2:], b[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area_a[:, None] + area_b - inter
        return inter / (union + 1e-6)

    stats = defaultdict(lambda: {"gt": 0, "matched": 0, "ious": []})
    model.eval()

    for img_path, graph_path, _ in tqdm(dataset.samples, desc="Class-wise Recall"):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        with open(graph_path, "r") as f:
            graph = json.load(f)

        # -------- GT (clamp + drop invalid) --------
        gt_boxes, gt_labels = [], []
        for node in graph.get("nodes", []):
            bbox = node.get("bbox")
            if not bbox:
                continue

            raw = node.get("data_class", "") or node.get("category", "")
            l2 = dataset.map_raw_to_l2(raw)
            if l2 not in dataset.label2id:
                continue

            clamped = clamp_bbox_xywh(bbox, w, h)
            if clamped is None:
                continue

            x, y, bw, bh = clamped
            gt_boxes.append([x, y, x + bw, y + bh])
            gt_labels.append(dataset.label2id[l2])
            stats[l2]["gt"] += 1

        if not gt_boxes:
            continue

        gt_boxes = torch.tensor(gt_boxes, device=device, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, device=device, dtype=torch.long)

        # -------- Preds --------
        post = detector_predict_post(model, dataset, img, device, score_thresh=float(score_thresh))

        if post["boxes"].numel() == 0:
            continue

        # Clip preds to bounds + drop degenerate
        pred_boxes, pred_scores, pred_labels = _clip_and_filter_xyxy(
            post["boxes"], post["scores"], post["labels"], w, h
        )
        if pred_boxes.numel() == 0:
            continue

        # Optional top-k by score
        if topk is not None:
            k = min(int(topk), int(pred_scores.numel()))
            keep = torch.argsort(pred_scores, descending=True)[:k]
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

        # -------- Matching --------
        ious = box_iou(gt_boxes, pred_boxes)

        for gi in range(len(gt_boxes)):
            gt_label = int(gt_labels[gi].item())
            gt_name = dataset.id2label[gt_label]

            same_class = (pred_labels == gt_label)
            if same_class.sum().item() == 0:
                continue

            max_iou = float(ious[gi][same_class].max().item())
            if max_iou >= iou_thresh:
                stats[gt_name]["matched"] += 1
                stats[gt_name]["ious"].append(max_iou)

    summary = {
        "config": {
            "iou_thresh": float(iou_thresh),
            "score_thresh": float(score_thresh),
            "topk": int(topk) if topk is not None else None,
        },
        "per_class": {},
    }

    for cls in dataset.level2_classes:
        s = stats[cls]
        ious_arr = np.array(s["ious"], dtype=np.float32) if s["ious"] else np.array([], dtype=np.float32)

        if s["gt"] > 0:
            recall = s["matched"] / s["gt"]
        else:
            recall = 0.0

        if ious_arr.size > 0:
            mean_iou = float(ious_arr.mean())
            median_iou = float(np.median(ious_arr))
            iou_ge_070 = float((ious_arr >= 0.70).mean())
            iou_ge_085 = float((ious_arr >= 0.85).mean())
        else:
            mean_iou = None
            median_iou = None
            iou_ge_070 = None
            iou_ge_085 = None

        summary["per_class"][cls] = {
            "gt": int(s["gt"]),
            "matched": int(s["matched"]),
            "recall": float(recall),
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "iou_ge_070": iou_ge_070,
            "iou_ge_085": iou_ge_085,
        }

    # -------- Report --------
    print("\n=== Class-wise Recall & IoU Diagnostics ===")
    for cls in dataset.level2_classes:
        s = stats[cls]
        if s["gt"] == 0:
            continue

        recall = summary["per_class"][cls]["recall"]
        ious_arr = np.array(s["ious"], dtype=np.float32) if s["ious"] else np.array([], dtype=np.float32)

        print(f"\n[{cls}]")
        print(f"  GT count        : {s['gt']}")
        print(f"  Recall@{iou_thresh:.2f}      : {recall:.3f}")

        if ious_arr.size > 0:
            print(f"  Mean IoU        : {ious_arr.mean():.3f}")
            print(f"  Median IoU      : {np.median(ious_arr):.3f}")
            print(f"  IoU ≥ 0.70      : {(ious_arr >= 0.70).mean():.3f}")
            print(f"  IoU ≥ 0.85      : {(ious_arr >= 0.85).mean():.3f}")
        else:
            print("  IoU stats       : none matched")

    return summary


def compute_area_weighted_recall(
    model,
    dataset,
    device,
    iou_thr: float = 0.5,
    score_thresh: float = 0.01,
    topk: int | None = 100,
    per_class: bool = False,
):
    """
    Area-weighted recall (class-aware, label-space-aware):

    - Only counts GT boxes that map into dataset.label2id (your 6 classes).
    - For each GT box, finds best IoU among *predictions of the same class*.
    - Adds the GT area to matched_area if best IoU >= iou_thr.
    - Optionally limits preds to top-k by score (COCO-like maxDets).

    Returns:
      - if per_class=False: float awr
      - if per_class=True : (overall_awr, dict[class_name -> awr])
    """
    model.eval()

    total_area = 0.0
    matched_area = 0.0

    # Optional per-class breakdown
    cls_total = {c: 0.0 for c in dataset.level2_classes}
    cls_matched = {c: 0.0 for c in dataset.level2_classes}

    for img_path, graph_path, _ in tqdm(dataset.samples, desc="Area-Weighted Recall"):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # ---- Predict ----
        post = detector_predict_post(model, dataset, img, device, score_thresh=float(score_thresh))

        if post["boxes"].numel() == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_labels = np.zeros((0,), dtype=np.int64)
            pred_scores = np.zeros((0,), dtype=np.float32)
        else:
            pb, ps, pl = _clip_and_filter_xyxy(post["boxes"], post["scores"], post["labels"], w, h)

            if pb.numel() == 0:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_labels = np.zeros((0,), dtype=np.int64)
                pred_scores = np.zeros((0,), dtype=np.float32)
            else:
                # top-k by score (COCO-like)
                if topk is not None:
                    k = min(int(topk), int(ps.numel()))
                    keep = torch.argsort(ps, descending=True)[:k]
                    pb = pb[keep]
                    ps = ps[keep]
                    pl = pl[keep]

                pred_boxes = pb.detach().cpu().numpy()
                pred_labels = pl.detach().cpu().numpy()
                pred_scores = ps.detach().cpu().numpy()  # not used, but handy for debugging

        # ---- Load GT ----
        with open(graph_path, "r") as f:
            gt = json.load(f)

        for node in gt.get("nodes", []):
            bbox = node.get("bbox")
            if not bbox:
                continue

            # Filter GT to your label space (critical fix #1)
            raw = node.get("data_class", "") or node.get("category", "")
            l2 = dataset.map_raw_to_l2(raw)
            if l2 not in dataset.label2id:
                continue

            clamped = clamp_bbox_xywh(bbox, w, h)
            if clamped is None:
                continue

            x, y, bw, bh = clamped
            area = float(bw * bh)

            gt_box = [x, y, x + bw, y + bh]  # xyxy

            total_area += area
            cls_total[l2] += area

            # Class-aware matching (critical fix #2)
            gt_lab = dataset.label2id[l2]
            same = (pred_labels == gt_lab)
            if same.sum() == 0:
                continue

            best_iou = 0.0
            for pb in pred_boxes[same]:
                best_iou = max(best_iou, float(compute_iou(gt_box, pb)))

            if best_iou >= iou_thr:
                matched_area += area
                cls_matched[l2] += area

    overall = matched_area / max(total_area, 1e-6)

    if not per_class:
        return overall

    per_cls = {c: (cls_matched[c] / max(cls_total[c], 1e-6)) for c in dataset.level2_classes}
    return overall, per_cls
