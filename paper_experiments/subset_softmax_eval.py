from utils.paper_io import table_path
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoImageProcessor, RTDetrForObjectDetection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as config_module
import hierarchy_config as hier
from data.dataset import GraphRTDetrDataset
from paper_experiments.common import _clip_and_filter_xyxy, apply_policy


def _resolve_split_list(path: str, base_dir: str) -> List[str]:
    roots = {
        "colorful": os.path.join(base_dir, "colorful"),
        "high_quality": os.path.join(base_dir, "high_quality"),
        "high_quality_architectural": os.path.join(base_dir, "high_quality_architectural"),
    }
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith("/"):
                line = line[:-1]
            if line.startswith("/"):
                line = line[1:]
            parts = line.split("/")
            dataset = parts[0]
            rest = "/".join(parts[1:])
            root = roots.get(dataset)
            if not root:
                continue
            full = os.path.join(root, rest)
            if os.path.isdir(full):
                out.append(full)
    return out


def _set_config_from_run(cfg: Dict):
    for key in (
        "BASE_DIR",
        "IMAGE_FILENAME",
        "MAX_NODES",
        "MIN_MAPPED_RATIO",
        "DISTANCE_SCORE_FILE",
    ):
        if key in cfg:
            setattr(config_module, key, cfg[key])


def _build_processor(cfg: Dict):
    processor = AutoImageProcessor.from_pretrained(cfg["BACKBONE"])
    processor.do_resize = cfg.get("RESIZE_ENABLE", True)
    if cfg.get("RESIZE_FIXED", True):
        processor.size = {
            "height": cfg.get("RESIZE_HEIGHT", 1024),
            "width": cfg.get("RESIZE_WIDTH", 1024),
        }
    else:
        processor.size = {
            "shortest_edge": cfg.get("RESIZE_SHORTEST_EDGE", 1024),
            "longest_edge": cfg.get("RESIZE_LONGEST_EDGE", 1024),
        }
    processor.do_pad = cfg.get("PAD_ENABLE", False)
    processor.pad_size = (
        {"height": cfg.get("PAD_SIZE", 1024), "width": cfg.get("PAD_SIZE", 1024)}
        if cfg.get("PAD_ENABLE", False)
        else None
    )
    return processor


def _subset_softmax_predictions(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    subset_ids: List[int],
    id2label: Dict[int, str],
    score_thresh: float,
    topk_pre: int,
    final_k: int,
):
    w, h = image.size
    enc = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**enc)

    logits = outputs.logits[0]  # [Q, C+1]
    pred_boxes = outputs.pred_boxes[0]  # [Q, 4] cxcywh in [0,1]

    bg_idx = logits.shape[-1] - 1
    subset_idx = list(subset_ids)
    logits_subset = torch.cat([logits[:, subset_idx], logits[:, [bg_idx]]], dim=-1)
    probs = torch.softmax(logits_subset, dim=-1)
    class_probs = probs[:, : len(subset_idx)]
    scores, best = class_probs.max(dim=-1)
    labels = torch.tensor([subset_idx[i] for i in best.tolist()], device=logits.device)

    # cxcywh -> xyxy in pixels
    cx, cy, bw, bh = pred_boxes.unbind(-1)
    x1 = (cx - 0.5 * bw) * w
    y1 = (cy - 0.5 * bh) * h
    x2 = (cx + 0.5 * bw) * w
    y2 = (cy + 0.5 * bh) * h
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    boxes, scores, labels = _clip_and_filter_xyxy(boxes, scores, labels, w, h)
    boxes, scores, labels = apply_policy(
        boxes,
        scores,
        labels,
        id2label=id2label,
        score_thresh=score_thresh,
        topk_pre=topk_pre,
        final_k=final_k,
    )
    return boxes, scores, labels


def _per_class_ap(eval_obj: COCOeval, class_names: List[str], cat_ids: List[int]):
    precision = eval_obj.eval["precision"]  # [T, R, K, A, M]
    iou_thrs = eval_obj.params.iouThrs
    ap_by_class = {}
    for k, (name, cid) in enumerate(zip(class_names, cat_ids)):
        ap_all = precision[:, :, k, 0, -1]
        ap_all = ap_all[ap_all > -1]
        ap = float(np.mean(ap_all)) if ap_all.size else float("nan")
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
        description="Subset-softmax COCO eval (no retraining)."
    )
    parser.add_argument("--run-dir", required=True, help="Run dir containing config.json and metrics/")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (defaults to run-dir/checkpoints/best)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--epoch", type=int, default=1, help="Epoch for val split (ann file).")
    parser.add_argument("--classes", nargs="+", default=["WINDOW", "DOOR"])
    parser.add_argument("--score-thresh", type=float, default=0.02)
    parser.add_argument("--topk-pre", type=int, default=150)
    parser.add_argument("--final-k", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--limit", type=int, default=0, help="Limit images for small validation runs.")
    args = parser.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    _set_config_from_run(cfg)

    if args.ckpt is None:
        args.ckpt = os.path.join(args.run_dir, "checkpoints", "best")

    if not args.out:
        run_name = Path(args.run_dir).name
        split_name = args.split
        classes_tag = "_".join(c.lower() for c in args.classes)
        args.out = str(table_path(f"table_subset_softmax_{run_name}_{split_name}_{classes_tag}.json"))

    processor = _build_processor(cfg)

    # Build split list (folders)
    split_txt = cfg["TEST_TXT"] if args.split == "test" else cfg["VAL_TXT"]
    folders = _resolve_split_list(split_txt, cfg["BASE_DIR"])
    dataset = GraphRTDetrDataset(folders, processor, os.path.join(args.run_dir, "hierarchy_config.py"), augment=False)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = RTDetrForObjectDetection.from_pretrained(args.ckpt).to(device).eval()

    # Subset ids
    label2id = dataset.label2id
    id2label = dataset.id2label
    subset_ids = [label2id[c] for c in args.classes]

    # COCO ann
    if args.split == "test":
        ann_path = os.path.join(args.run_dir, "metrics", "test_ann.json")
    else:
        ann_path = os.path.join(args.run_dir, "metrics", f"val_ann_epoch_{args.epoch:03d}.json")
        if not os.path.exists(ann_path):
            # fallback to first available val ann file
            candidates = sorted(glob.glob(os.path.join(args.run_dir, "metrics", "val_ann_epoch_*.json")))
            if candidates:
                ann_path = candidates[0]

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    # Predict
    preds = []
    samples = dataset.samples[: args.limit] if args.limit and args.limit > 0 else dataset.samples
    for idx, (img_path, _, _) in enumerate(tqdm(samples, desc="Subset-softmax preds")):
        img = Image.open(img_path).convert("RGB")
        boxes, scores, labels = _subset_softmax_predictions(
            model,
            processor,
            img,
            device,
            subset_ids=subset_ids,
            id2label=id2label,
            score_thresh=args.score_thresh,
            topk_pre=args.topk_pre,
            final_k=args.final_k,
        )
        if boxes.numel() == 0:
            continue
        for b, s, l in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b.tolist()
            preds.append(
                {
                    "image_id": idx,
                    "category_id": int(l.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s.item()),
                }
            )

    # Evaluate
    coco_gt = COCO(ann_path)
    coco_dt = coco_gt.loadRes(preds)
    cat_ids = [c["id"] for c in coco_gt.dataset.get("categories", []) if c["name"] in args.classes]
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = cat_ids
    coco_eval.params.imgIds = coco_gt.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "run_dir": args.run_dir,
        "ckpt": args.ckpt,
        "split": args.split,
        "epoch": args.epoch if args.split == "val" else None,
        "classes": args.classes,
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

    print("\n=== Subset-softmax COCO stats (classes: {}) ===".format(", ".join(args.classes)))
    for k, v in stats["coco_stats"].items():
        print(f"{k:>9}: {v:.4f}")

    print("\n=== Per-class AP ===")
    for cls, vals in stats["per_class_ap"].items():
        print(
            f"{cls:>7}: AP={vals['ap']:.4f}  AP50={vals['ap50']:.4f}  AP75={vals['ap75']:.4f}"
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved JSON: {args.out}")


if __name__ == "__main__":
    main()
