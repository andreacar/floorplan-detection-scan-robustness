#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_experiments.common import (
    DEFAULT_PER_CLASS_CAP,
    infer_predictions,
    load_gt_from_graph,
    load_image,
    load_label_maps,
    load_model,
    load_test_dirs,
    match_greedy_by_class,
    safe_makedirs,
)


def _parse_runs(run_args):
    runs = []
    for spec in run_args:
        parts = spec.split(",")
        run = {}
        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            run[k.strip()] = v.strip()
        if "name" in run and "ckpt" in run and "image" in run:
            runs.append(run)
    return runs


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    x_t = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = x_t * w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(w.item()), float(b.item())


def _bin_stats(x: np.ndarray, y: np.ndarray, bins: int = 15):
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo, hi = float(x.min()), float(x.max())
    edges = np.linspace(lo, hi, bins + 1)
    centers = []
    means = []
    counts = []
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < len(edges) - 2 else x <= edges[i + 1])
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(float(np.mean(y[mask])))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(means), np.array(counts)


def main():
    parser = argparse.ArgumentParser(description="Success vs object size with logistic fit.")
    parser.add_argument("--run", action="append", default=[], help="name=clean,ckpt=...,image=...")
    parser.add_argument("--ckpt", default=None, help="Single-run checkpoint.")
    parser.add_argument("--image-name", default=None, help="Single-run image name.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--out-dir", default="paper_experiments/out/size_success", help="Output dir.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Score threshold.")
    parser.add_argument("--topk-pre", type=int, default=0, help="Pre top-k by score.")
    parser.add_argument("--final-k", type=int, default=0, help="Final per-image cap.")
    parser.add_argument("--use-per-class-thresh", action="store_true", help="Use per-class thresholds.")

    args = parser.parse_args()

    safe_makedirs(args.out_dir)
    device = torch.device(args.device)

    runs = _parse_runs(args.run)
    if not runs:
        if not args.ckpt or not args.image_name:
            raise RuntimeError("Provide --run or both --ckpt and --image-name.")
        runs = [{"name": "run", "ckpt": args.ckpt, "image": args.image_name}]

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    test_dirs = load_test_dirs(args.test_txt)
    if args.limit and args.limit > 0:
        test_dirs = test_dirs[: args.limit]

    per_class_cap = DEFAULT_PER_CLASS_CAP.copy()
    all_results = {}

    for run in runs:
        processor, model = load_model(run["ckpt"], device)
        areas = []
        success50 = []
        success85 = []

        for folder in test_dirs:
            img_path = os.path.join(folder, run["image"])
            graph_path = os.path.join(folder, "graph.json")
            if not (os.path.exists(img_path) and os.path.exists(graph_path)):
                continue

            img = load_image(img_path)
            gt_boxes, gt_labels, gt_areas = load_gt_from_graph(
                graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
            )
            if not gt_boxes:
                continue

            boxes, scores, labels = infer_predictions(
                model,
                processor,
                img,
                device,
                id2label,
                score_thresh=args.score_thresh,
                topk_pre=args.topk_pre,
                final_k=args.final_k,
                per_class_cap=per_class_cap,
                use_per_class_thresh=args.use_per_class_thresh,
            )

            matched_iou = match_greedy_by_class(
                gt_boxes,
                gt_labels,
                boxes.tolist(),
                scores.tolist(),
                labels.tolist(),
                iou_thr=0.5,
            )
            for area, iou_val in zip(gt_areas, matched_iou):
                areas.append(float(area))
                success50.append(1.0 if iou_val >= 0.5 else 0.0)
                success85.append(1.0 if iou_val >= 0.85 else 0.0)

        areas_np = np.array(areas, dtype=float)
        log_area = np.log(np.maximum(areas_np, 1e-6))
        s50 = np.array(success50, dtype=float)
        s85 = np.array(success85, dtype=float)

        w50, b50 = _fit_logistic(log_area, s50)
        w85, b85 = _fit_logistic(log_area, s85)
        area50 = float(math.exp(-b50 / w50)) if abs(w50) > 1e-9 else float("nan")
        area85 = float(math.exp(-b85 / w85)) if abs(w85) > 1e-9 else float("nan")

        centers50, means50, counts50 = _bin_stats(log_area, s50, bins=15)
        centers85, means85, counts85 = _bin_stats(log_area, s85, bins=15)

        result = {
            "count": int(len(areas)),
            "logistic_50": {"w": w50, "b": b50, "area_at_50pct": area50},
            "logistic_85": {"w": w85, "b": b85, "area_at_50pct": area85},
            "bins_50": {"centers": centers50.tolist(), "means": means50.tolist(), "counts": counts50.tolist()},
            "bins_85": {"centers": centers85.tolist(), "means": means85.tolist(), "counts": counts85.tolist()},
        }
        all_results[run["name"]] = result
        out_path = os.path.join(args.out_dir, f"{run['name']}_size_success.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot success vs size
    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#f0e442"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for idx, run in enumerate(runs):
        res = all_results[run["name"]]
        color = colors[idx % len(colors)]
        centers = np.array(res["bins_50"]["centers"])
        means = np.array(res["bins_50"]["means"])
        if centers.size > 0:
            axes[0].plot(centers, means, marker="o", color=color, label=run["name"])
        w, b = res["logistic_50"]["w"], res["logistic_50"]["b"]
        if not math.isnan(w) and abs(w) > 1e-9:
            xs = np.linspace(centers.min() if centers.size else -5, centers.max() if centers.size else 5, 200)
            ys = 1.0 / (1.0 + np.exp(-(w * xs + b)))
            axes[0].plot(xs, ys, color=color, alpha=0.6)

        centers = np.array(res["bins_85"]["centers"])
        means = np.array(res["bins_85"]["means"])
        if centers.size > 0:
            axes[1].plot(centers, means, marker="o", color=color, label=run["name"])
        w, b = res["logistic_85"]["w"], res["logistic_85"]["b"]
        if not math.isnan(w) and abs(w) > 1e-9:
            xs = np.linspace(centers.min() if centers.size else -5, centers.max() if centers.size else 5, 200)
            ys = 1.0 / (1.0 + np.exp(-(w * xs + b)))
            axes[1].plot(xs, ys, color=color, alpha=0.6)

    axes[0].set_title("Success vs log(area) @ IoU>=0.50")
    axes[1].set_title("Success vs log(area) @ IoU>=0.85")
    axes[0].set_xlabel("log(area)")
    axes[1].set_xlabel("log(area)")
    axes[0].set_ylabel("Success rate")
    axes[0].legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "size_success.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved summary -> {summary_path}")
    print(f"[OK] Saved figure  -> {fig_path}")


if __name__ == "__main__":
    main()
