#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import argparse
from collections import Counter

from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoImageProcessor, RTDetrForObjectDetection
from utils.paths import load_split_list
from data.dataset import GraphRTDetrDataset
from utils.geometry import compute_iou
from data.coco_utils import build_coco_groundtruth
from pycocotools.coco import COCO

from config import CKPT_DIR, VAL_TXT, CLASS_COLORS, MET_DIR


def _label_name(ds, lab_int: int) -> str:
    if hasattr(ds, "id2label") and isinstance(ds.id2label, dict):
        return ds.id2label.get(lab_int, str(lab_int))
    return str(lab_int)


def _draw_ranked(img: Image.Image, boxes, labels, scores, ds, topk: int, out_path: str):
    im = img.copy()
    dr = ImageDraw.Draw(im)

    k = min(topk, len(scores))
    for r in range(k):
        x1, y1, x2, y2 = boxes[r]
        lab = int(labels[r])
        score = float(scores[r])
        name = _label_name(ds, lab)
        color = CLASS_COLORS.get(name, (255, 0, 0))

        dr.rectangle([x1, y1, x2, y2], outline=color, width=3)
        dr.text((x1 + 2, y1 + 2), f"#{r+1} {name} {score:.3f}", fill=color)

    im.save(out_path)


def _ensure_coco_gt(ds, ann_path: str):
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    if not os.path.exists(ann_path):
        build_coco_groundtruth(ds, ann_path)


def _load_coco_gt_for_image(ann_path: str, image_id: int):
    coco = COCO(ann_path)
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)

    gt_boxes = []
    gt_labels = []
    for a in anns:
        x, y, w, h = a["bbox"]
        gt_boxes.append([float(x), float(y), float(x + w), float(y + h)])
        gt_labels.append(int(a["category_id"]))
    return gt_boxes, gt_labels


def _match_greedy_classaware(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh: float, topk: int):
    matched_gt = set()
    per_class = Counter()
    matched_pred_ranks = []

    K = min(topk, len(pred_boxes))
    for r in range(K):
        pbox = pred_boxes[r]
        plab = int(pred_labels[r])

        best_iou = 0.0
        best_j = None

        for j, (gbox, glab) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched_gt:
                continue
            if int(glab) != plab:
                continue
            iou = compute_iou(gbox, pbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None and best_iou >= iou_thresh:
            matched_gt.add(best_j)
            per_class[plab] += 1
            matched_pred_ranks.append(r + 1)

    matched_pred_ranks.sort()
    return matched_pred_ranks, matched_gt, per_class


def _knee_cutoff(scores, min_keep=30, max_keep=100):
    s = np.asarray(scores, dtype=np.float32)
    n = s.shape[0]
    if n == 0:
        return 0
    if n <= 3:
        return int(np.clip(n, min_keep, max_keep))

    smin, smax = float(s.min()), float(s.max())
    denom = (smax - smin) if (smax - smin) > 1e-9 else 1.0
    y = (s - smin) / denom

    x = np.linspace(0.0, 1.0, n)
    y_line = 1.0 - x

    dist = np.abs(y - y_line)
    knee_idx = int(dist.argmax())
    K = knee_idx + 1
    return int(np.clip(K, min_keep, min(max_keep, n)))


def _drop_ratio_cutoff(scores, drop_ratio=0.70, min_keep=30, max_keep=100):
    """
    Keep until the first big local score drop:
      score[r] / score[r-1] < drop_ratio
    If no big drop, keep max_keep.
    """
    s = np.asarray(scores, dtype=np.float32)
    n = s.shape[0]
    if n == 0:
        return 0

    max_keep = min(max_keep, n)
    if n <= 2:
        return int(np.clip(n, min_keep, max_keep))

    K = max_keep
    for r in range(1, max_keep):
        prev = float(s[r - 1])
        cur = float(s[r])
        if prev <= 1e-9:
            continue
        if (cur / prev) < drop_ratio:
            K = r  # keep up to r (1-indexed would be r, but K is count)
            break

    return int(np.clip(K, min_keep, max_keep))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--tops", type=str, default="10,30,50,100")
    ap.add_argument("--out_dir", type=str, default="./outputs/diagnostic_rankings")
    ap.add_argument("--print_top", type=int, default=30)
    ap.add_argument("--iou", type=float, default=0.5)

    ap.add_argument("--min_keep", type=int, default=30)
    ap.add_argument("--max_keep", type=int, default=100)

    ap.add_argument("--drop_ratio", type=float, default=0.70)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(CKPT_DIR, "best")
    os.makedirs(args.out_dir, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=False)
    model = RTDetrForObjectDetection.from_pretrained(ckpt).to(device).eval()

    val_dirs = load_split_list(VAL_TXT)
    ds = GraphRTDetrDataset(val_dirs, processor, "hierarchy_config.py", augment=False)

    img_path, _, _ = ds.samples[args.idx]
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    enc = ds.image_processor(images=img, return_tensors="pt").to(device)
    out = model(**enc)

    post = ds.image_processor.post_process_object_detection(
        out,
        threshold=args.thr,
        target_sizes=torch.tensor([[h, w]], device=device),
    )[0]

    scores_t = post["scores"].detach().cpu()
    order = torch.argsort(scores_t, descending=True)

    boxes = post["boxes"].detach().cpu()[order].tolist()
    labels = post["labels"].detach().cpu()[order].tolist()
    scores = scores_t[order].tolist()

    c = Counter()
    for lab in labels:
        c[_label_name(ds, int(lab))] += 1

    print(f"\nIDX={args.idx} thr={args.thr} total={len(scores)}")
    print("per-class:", dict(c))
    print("image:", img_path)

    K_knee = _knee_cutoff(scores, min_keep=args.min_keep, max_keep=args.max_keep)
    K_drop = _drop_ratio_cutoff(scores, drop_ratio=args.drop_ratio, min_keep=args.min_keep, max_keep=args.max_keep)

    print(f"\nKNEE cutoff suggests K={K_knee} (min_keep={args.min_keep}, max_keep={args.max_keep})")
    print(f"DROP cutoff suggests K={K_drop} (drop_ratio={args.drop_ratio}, min_keep={args.min_keep}, max_keep={args.max_keep})")

    nprint = min(args.print_top, len(scores))
    print(f"\nTop-{nprint} ranked predictions:")
    for r in range(nprint):
        name = _label_name(ds, int(labels[r]))
        print(f"#{r+1:03d} score={scores[r]:.4f} label={name} box={boxes[r]}")

    tops = [int(x.strip()) for x in args.tops.split(",") if x.strip()]
    for k in tops:
        out_path = os.path.join(args.out_dir, f"idx{args.idx}_thr{args.thr}_rank_top{k}.png")
        _draw_ranked(img, boxes, labels, scores, ds, topk=k, out_path=out_path)
        print("saved:", out_path)

    knee_path = os.path.join(args.out_dir, f"idx{args.idx}_thr{args.thr}_rank_topKnee{K_knee}.png")
    _draw_ranked(img, boxes, labels, scores, ds, topk=K_knee, out_path=knee_path)
    print("saved:", knee_path)

    drop_path = os.path.join(args.out_dir, f"idx{args.idx}_thr{args.thr}_rank_topDrop{K_drop}.png")
    _draw_ranked(img, boxes, labels, scores, ds, topk=K_drop, out_path=drop_path)
    print("saved:", drop_path)

    # score-vs-rank plot + vertical lines
    if len(scores) > 0:
        plt.figure()
        xs = list(range(1, len(scores) + 1))
        plt.plot(xs, scores)
        plt.axvline(K_knee, linestyle="--")
        plt.axvline(K_drop, linestyle=":")
        plt.xlabel("Rank (1 = highest score)")
        plt.ylabel("Score")
        plt.title(f"Scores vs Rank idx={args.idx} | knee={K_knee}, drop={K_drop}")
        plot_path = os.path.join(args.out_dir, f"idx{args.idx}_thr{args.thr}_scores_vs_rank.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("saved:", plot_path)

    # COCO-GT recall checks
    ann_path = os.path.join(MET_DIR, "val_ann.json")
    _ensure_coco_gt(ds, ann_path)
    gt_boxes, gt_labels = _load_coco_gt_for_image(ann_path, image_id=int(args.idx))
    print(f"\nCOCO-GT boxes: {len(gt_boxes)} (from {ann_path})")

    iou_thresh = float(args.iou)
    for K in [10, 30, 50, K_drop, 100]:
        matched_ranks, matched_gt, _ = _match_greedy_classaware(
            pred_boxes=boxes,
            pred_labels=labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_thresh=iou_thresh,
            topk=int(K),
        )
        rec = len(matched_gt) / max(len(gt_boxes), 1)
        print(f"[COCO-GT] Recall within top-{int(K)} @IoU{int(iou_thresh*100)}: {len(matched_gt)}/{len(gt_boxes)} = {rec:.3f}")


if __name__ == "__main__":
    main()
