#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import sys

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


def _init_counts(classes):
    return {c: {"total": 0, "missed": 0, "loose": 0, "tight": 0} for c in classes}


def main():
    parser = argparse.ArgumentParser(description="Error decomposition: missed vs loose vs tight.")
    parser.add_argument("--run", action="append", default=[], help="name=clean,ckpt=...,image=...")
    parser.add_argument("--ckpt", default=None, help="Single-run checkpoint.")
    parser.add_argument("--image-name", default=None, help="Single-run image name.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--out-dir", default="paper_experiments/out/error_decomp", help="Output dir.")
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
        counts = _init_counts(classes)
        overall = {"total": 0, "missed": 0, "loose": 0, "tight": 0}

        for folder in test_dirs:
            img_path = os.path.join(folder, run["image"])
            graph_path = os.path.join(folder, "graph.json")
            if not (os.path.exists(img_path) and os.path.exists(graph_path)):
                continue

            img = load_image(img_path)
            gt_boxes, gt_labels, _ = load_gt_from_graph(
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
            for i, iou_val in enumerate(matched_iou):
                cls_name = classes[gt_labels[i]]
                counts[cls_name]["total"] += 1
                overall["total"] += 1
                if iou_val < 0.5:
                    counts[cls_name]["missed"] += 1
                    overall["missed"] += 1
                elif iou_val < 0.85:
                    counts[cls_name]["loose"] += 1
                    overall["loose"] += 1
                else:
                    counts[cls_name]["tight"] += 1
                    overall["tight"] += 1

        result = {"counts": counts, "overall": overall}
        all_results[run["name"]] = result
        out_path = os.path.join(args.out_dir, f"{run['name']}_decomp.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot per-run stacked bars
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4), sharey=True)
    if n_runs == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        res = all_results[run["name"]]["counts"]
        totals = np.array([res[c]["total"] for c in classes], dtype=float)
        missed = np.array([res[c]["missed"] for c in classes], dtype=float)
        loose = np.array([res[c]["loose"] for c in classes], dtype=float)
        tight = np.array([res[c]["tight"] for c in classes], dtype=float)
        denom = np.maximum(totals, 1.0)
        missed_p = missed / denom
        loose_p = loose / denom
        tight_p = tight / denom
        x = np.arange(len(classes))
        ax.bar(x, missed_p, label="missed", color="#d55e00")
        ax.bar(x, loose_p, bottom=missed_p, label="loose", color="#f0e442")
        ax.bar(x, tight_p, bottom=missed_p + loose_p, label="tight", color="#009e73")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(run["name"])
        ax.set_ylabel("Fraction of GT")
    axes[0].legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "error_decomposition.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
