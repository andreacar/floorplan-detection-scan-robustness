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
from typing import Sequence, Tuple

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    IsotonicRegression = None
    LogisticRegression = None
    average_precision_score = None
    brier_score_loss = None
    roc_auc_score = None
    StratifiedKFold = None

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
    pearson_corr,
    safe_makedirs,
    spearman_corr,
)
from utils.geometry import clamp_bbox_xywh


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    comp = vh[:2].T
    return x0 @ comp


FAILURE_RECALL_THRESH = 0.2
PCA_LOW_Q = 0.10
PCA_HIGH_Q = 0.90
PCA_MIN_SAMPLES = 30
PCA_MAX_POINTS = 150


def _confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    n = values.size
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    z = 1.96 if confidence >= 0.95 else 1.0
    return mean - z * se, mean + z * se


def _bin_stats(x: np.ndarray, y: np.ndarray, bins: int = 10):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    centers, mean_vals, ci_lows, ci_highs, counts = [], [], [], [], []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (lo + hi))
        values = y[mask]
        ci_low, ci_high = _confidence_interval(values)
        mean_vals.append(float(np.mean(values)))
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        counts.append(int(mask.sum()))
    return (
        np.array(centers),
        np.array(mean_vals),
        np.array(ci_lows),
        np.array(ci_highs),
        np.array(counts),
    )


def _bin_failure_rate(x: np.ndarray, failure_mask: np.ndarray, bins: int = 10):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.array([]), np.array([])
    centers, rates = [], []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (lo + hi))
        rates.append(float(failure_mask[mask].mean()))
    return np.array(centers), np.array(rates)


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, samples: int = 1000) -> Tuple[float, float, float]:
    rng = np.random.default_rng(42)
    n = x.size
    if n < 2:
        return 0.0, 0.0, 0.0
    stats = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        stats.append(spearman_corr(x[idx], y[idx]))
    stats = np.sort(np.array(stats))
    median = float(np.median(stats))
    lo = float(np.quantile(stats, 0.025))
    hi = float(np.quantile(stats, 0.975))
    return median, lo, hi


def _sample_indices(mask: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.nonzero(mask)[0]
    if idx.size <= max_count:
        return idx
    return rng.choice(idx, size=max_count, replace=False)


def _plot_pca_panel(
    ax,
    emb_a_subset: np.ndarray,
    emb_b_subset: np.ndarray,
    delta_subset: np.ndarray,
    title: str,
    cmap: str,
    show_colorbar: bool,
):
    if emb_a_subset.size == 0 or emb_b_subset.size == 0:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", fontsize=10, color="0.5")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    stacked = np.vstack([emb_a_subset, emb_b_subset])
    pca_data = _pca_2d(stacked)
    pca_a = pca_data[: emb_a_subset.shape[0]]
    pca_b = pca_data[emb_a_subset.shape[0] :]

    for i in range(pca_a.shape[0]):
        ax.plot([pca_a[i, 0], pca_b[i, 0]], [pca_a[i, 1], pca_b[i, 1]], color="0.85", linewidth=0.6)
    ax.scatter(pca_a[:, 0], pca_a[:, 1], s=18, color="0.4", label="Clean")
    sc = ax.scatter(pca_b[:, 0], pca_b[:, 1], s=35, c=delta_subset, cmap=cmap, label="Scanned", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", fontsize=8)
    if show_colorbar:
        cb = plt.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("ΔAP@0.85")


def _ensure_min_mask(mask: np.ndarray, fallback: np.ndarray, min_count: int) -> np.ndarray:
    if mask.sum() >= min_count:
        return mask
    if fallback.sum() >= min_count:
        return fallback
    if mask.sum() > 0:
        return mask
    return fallback


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


def _save_pair_gallery(
    records: Sequence[dict],
    indices: np.ndarray,
    image_a: str,
    image_b: str,
    out_dir: str,
    prefix: str,
    title: str,
):
    if not indices.size:
        return
    fig, axs = plt.subplots(2, len(indices), figsize=(len(indices) * 2, 4))
    for col, idx in enumerate(indices):
        rec = records[idx]
        for row, img_name in enumerate((image_a, image_b)):
            ax = axs[row, col]
            img_path = os.path.join(rec["folder"], img_name)
            if os.path.exists(img_path):
                img = load_image(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            label = "Clean" if row == 0 else "Scanned"
            ax.set_title(label, fontsize=8)
            if row == 1:
                text = (
                    f"{rec['id']} | dist={rec['dist_l2']:.2f} | "
                    f"AP@0.85={rec['ap85_b']:.2f} | R@0.85={rec['recall85_b']:.2f}"
                )
                ax.text(0.5, -0.15, text, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _evaluate_logistic(X: np.ndarray, y: np.ndarray, name: str, summary: dict):
    if LogisticRegression is None:
        return None
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if roc_auc_score else None
    pr_auc = average_precision_score(y, probs) if average_precision_score else None
    brier = brier_score_loss(y, probs) if brier_score_loss else None
    summary[f"auc_{name}"] = auc
    summary[f"pr_auc_{name}"] = pr_auc
    summary[f"brier_{name}"] = brier
    k = max(1, int(np.round(0.1 * X.shape[0])))
    top_idx = np.argsort(probs)[-k:]
    top_rate = float(y[top_idx].mean())
    base_rate = float(y.mean())
    summary[f"top10_rate_{name}"] = top_rate
    summary[f"base_rate_{name}"] = base_rate
    summary[f"enrichment_{name}"] = top_rate / base_rate if base_rate > 0 else None
    return model, probs


def _stratified_cv_metrics(X: np.ndarray, y: np.ndarray, folds: int = 5, top_frac: float = 0.10):
    """
    Returns dict: metric -> (mean, ci_lo, ci_hi) using stratified CV.
    Robust to class imbalance; returns None if not enough positives/negatives.
    """
    if LogisticRegression is None or StratifiedKFold is None:
        return None

    y = np.asarray(y, dtype=int)
    pos = int(y.sum())
    neg = int((y == 0).sum())
    if pos < 2 or neg < 2:
        return None

    n_splits = min(int(folds), pos, neg)
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, prs, briers, enrichs, bases = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        yv = y[val_idx]

        if roc_auc_score is not None:
            aucs.append(float(roc_auc_score(yv, probs)))
        if average_precision_score is not None:
            prs.append(float(average_precision_score(yv, probs)))
        if brier_score_loss is not None:
            briers.append(float(brier_score_loss(yv, probs)))

        k = max(1, int(np.round(top_frac * val_idx.size)))
        top_idx = np.argsort(probs)[-k:]
        top_rate = float(yv[top_idx].mean())
        base_rate = float(yv.mean())
        bases.append(base_rate)
        enrichs.append(top_rate / base_rate if base_rate > 0 else np.nan)

    def _mean_ci(arr_list):
        arr = np.array(arr_list, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return (None, None, None)
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return (float(np.mean(arr)), float(lo), float(hi))

    return {
        "auc": _mean_ci(aucs),
        "pr": _mean_ci(prs),
        "brier": _mean_ci(briers),
        "enrich": _mean_ci(enrichs),
        "base": _mean_ci(bases),
        "folds_used": (float(n_splits), float(n_splits), float(n_splits)),
    }


def _failure_masks_from_quantiles(values: np.ndarray, quantiles: Sequence[float]):
    out = {}
    for q in quantiles:
        thr = float(np.quantile(values, q))
        out[q] = (values <= thr, thr)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Embedding distance vs per-image error + PCA visualization.",
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint path for RT-DETR.")
    parser.add_argument("--image-a", default="model_baked.png", help="Clean image name.")
    parser.add_argument("--image-b", default="F1_scaled.png", help="Scanned image name.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--out-dir", default="paper_experiments/out/embedding", help="Output dir.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Score threshold.")
    parser.add_argument("--topk-pre", type=int, default=0, help="Pre top-k by score.")
    parser.add_argument("--final-k", type=int, default=0, help="Final per-image cap.")
    parser.add_argument("--use-per-class-thresh", action="store_true", help="Use per-class thresholds.")
    parser.add_argument("--failure-quantile", type=float, default=0.2, help="Percentile for failure definition (Recall@0.85).")
    parser.add_argument("--risk-folds", type=int, default=5, help="Stratified folds for risk CV metrics.")

    args = parser.parse_args()

    safe_makedirs(args.out_dir)
    device = torch.device(args.device)

    level2, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(args.ckpt, device)

    test_dirs = load_test_dirs(args.test_txt)
    if args.limit and args.limit > 0:
        test_dirs = test_dirs[: args.limit]

    per_class_cap = DEFAULT_PER_CLASS_CAP.copy()

    records = []
    emb_a = []
    emb_b = []

    start_time = time.time()
    for idx, folder in enumerate(tqdm(test_dirs, desc="Pairs")):
        img_a_path = os.path.join(folder, args.image_a)
        img_b_path = os.path.join(folder, args.image_b)
        graph_path = os.path.join(folder, "graph.json")
        if not (os.path.exists(img_a_path) and os.path.exists(img_b_path) and os.path.exists(graph_path)):
            continue

        img_a = load_image(img_a_path)
        img_b = load_image(img_b_path)

        gt_boxes_a, gt_labels_a, _ = load_gt_from_graph(
            graph_path, img_a.size[0], img_a.size[1], map_raw_to_l2, label2id
        )
        gt_boxes_b, gt_labels_b, _ = load_gt_from_graph(
            graph_path, img_b.size[0], img_b.size[1], map_raw_to_l2, label2id
        )

        boxes_a, scores_a, labels_a = infer_predictions(
            model,
            processor,
            img_a,
            device,
            id2label,
            score_thresh=args.score_thresh,
            topk_pre=args.topk_pre,
            final_k=args.final_k,
            per_class_cap=per_class_cap,
            use_per_class_thresh=args.use_per_class_thresh,
        )
        boxes_b, scores_b, labels_b = infer_predictions(
            model,
            processor,
            img_b,
            device,
            id2label,
            score_thresh=args.score_thresh,
            topk_pre=args.topk_pre,
            final_k=args.final_k,
            per_class_cap=per_class_cap,
            use_per_class_thresh=args.use_per_class_thresh,
        )

        ap50_a = compute_ap_for_image(
            gt_boxes_a,
            gt_labels_a,
            boxes_a.tolist(),
            scores_a.tolist(),
            labels_a.tolist(),
            iou_thr=0.5,
            class_ids=label2id.values(),
        )
        ap50_b = compute_ap_for_image(
            gt_boxes_b,
            gt_labels_b,
            boxes_b.tolist(),
            scores_b.tolist(),
            labels_b.tolist(),
            iou_thr=0.5,
            class_ids=label2id.values(),
        )
        ap85_a = compute_ap_for_image(
            gt_boxes_a,
            gt_labels_a,
            boxes_a.tolist(),
            scores_a.tolist(),
            labels_a.tolist(),
            iou_thr=0.85,
            class_ids=label2id.values(),
        )
        ap85_b = compute_ap_for_image(
            gt_boxes_b,
            gt_labels_b,
            boxes_b.tolist(),
            scores_b.tolist(),
            labels_b.tolist(),
            iou_thr=0.85,
            class_ids=label2id.values(),
        )

        matched_iou_a = match_greedy_by_class(
            gt_boxes_a,
            gt_labels_a,
            boxes_a.tolist(),
            scores_a.tolist(),
            labels_a.tolist(),
            iou_thr=0.5,
        )
        matched_iou_b = match_greedy_by_class(
            gt_boxes_b,
            gt_labels_b,
            boxes_b.tolist(),
            scores_b.tolist(),
            labels_b.tolist(),
            iou_thr=0.5,
        )
        recall50_a = float(sum(i >= 0.5 for i in matched_iou_a)) / max(len(matched_iou_a), 1)
        recall50_b = float(sum(i >= 0.5 for i in matched_iou_b)) / max(len(matched_iou_b), 1)
        recall85_a = float(sum(i >= 0.85 for i in matched_iou_a)) / max(len(matched_iou_a), 1)
        recall85_b = float(sum(i >= 0.85 for i in matched_iou_b)) / max(len(matched_iou_b), 1)

        mask_clean = _build_roi_mask(graph_path, img_a.size[0], img_a.size[1])
        mask_scan = _build_roi_mask(graph_path, img_b.size[0], img_b.size[1])
        contrast_b = _estimate_contrast(img_b)
        roi_clean = _apply_mask(img_a, mask_clean, keep=True)
        roi_scan = _apply_mask(img_b, mask_scan, keep=True)
        bg_clean = _apply_mask(img_a, mask_clean, keep=False)
        bg_scan = _apply_mask(img_b, mask_scan, keep=False)

        emb_a_vec = extract_embedding(model, processor, img_a, device)
        emb_b_vec = extract_embedding(model, processor, img_b, device)
        emb_roi_a = extract_embedding(model, processor, roi_clean, device)
        emb_roi_b = extract_embedding(model, processor, roi_scan, device)
        emb_bg_a = extract_embedding(model, processor, bg_clean, device)
        emb_bg_b = extract_embedding(model, processor, bg_scan, device)
        emb_a.append(emb_a_vec)
        emb_b.append(emb_b_vec)

        l2_dist = float(np.linalg.norm(emb_a_vec - emb_b_vec))
        denom = float(np.linalg.norm(emb_a_vec) * np.linalg.norm(emb_b_vec) + 1e-9)
        cos_dist = float(1.0 - (float(np.dot(emb_a_vec, emb_b_vec)) / denom))
        dist_roi = float(np.linalg.norm(emb_roi_a - emb_roi_b))
        dist_bg = float(np.linalg.norm(emb_bg_a - emb_bg_b))
        mean_score_b = float(np.mean(scores_b)) if scores_b.size else 0.0
        num_detections_b = int(scores_b.size)

        records.append(
            {
                "id": os.path.basename(folder.rstrip("/")),
                "folder": folder,
                "ap50_a": ap50_a,
                "ap50_b": ap50_b,
                "ap85_a": ap85_a,
                "ap85_b": ap85_b,
                "recall50_a": recall50_a,
                "recall50_b": recall50_b,
                "recall85_a": recall85_a,
                "recall85_b": recall85_b,
                "delta_ap50": ap50_a - ap50_b,
                "delta_ap85": ap85_a - ap85_b,
                "delta_recall50": recall50_a - recall50_b,
                "delta_recall85": recall85_a - recall85_b,
                "dist_l2": l2_dist,
                "dist_cos": cos_dist,
                "mean_score_b": mean_score_b,
                "n_detections_b": num_detections_b,
                "contrast_b": contrast_b,
                "dist_roi": dist_roi,
                "dist_bg": dist_bg,
            }
        )

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - start_time
            per_item = elapsed / float(idx + 1)
            print(f"[progress] {idx + 1}/{len(test_dirs)} avg {per_item:.2f}s per pair")

    if not records:
        raise RuntimeError("No paired samples found. Check image names and test split.")

    total_time = time.time() - start_time
    if records:
        print(f"Finished {len(records)} pairs in {total_time:.1f}s ({total_time / len(records):.2f}s per pair)")

    emb_a = np.vstack(emb_a)
    emb_b = np.vstack(emb_b)

    dist_l2 = np.array([r["dist_l2"] for r in records])
    delta_ap50 = np.array([r["delta_ap50"] for r in records])
    delta_ap85 = np.array([r["delta_ap85"] for r in records])
    delta_recall50 = np.array([r["delta_recall50"] for r in records])
    delta_recall85 = np.array([r["delta_recall85"] for r in records])
    recall85_b = np.array([r["recall85_b"] for r in records])

    summary = {
        "count": len(records),
        "pearson_l2_delta_ap50": pearson_corr(dist_l2, delta_ap50),
        "pearson_l2_delta_ap85": pearson_corr(dist_l2, delta_ap85),
        "spearman_l2_delta_ap50": spearman_corr(dist_l2, delta_ap50),
        "spearman_l2_delta_ap85": spearman_corr(dist_l2, delta_ap85),
        "spearman_l2_delta_recall50": spearman_corr(dist_l2, delta_recall50),
        "spearman_l2_delta_recall85": spearman_corr(dist_l2, delta_recall85),
        "spearman_l2_recall85": spearman_corr(dist_l2, recall85_b),
    }

    mean_score_b = np.array([r["mean_score_b"] for r in records])
    n_detections_b = np.array([r["n_detections_b"] for r in records])
    contrast_b = np.array([r["contrast_b"] for r in records])
    dist_roi = np.array([r["dist_roi"] for r in records])
    dist_bg = np.array([r["dist_bg"] for r in records])

    summary["spearman_roi_recall85"] = spearman_corr(dist_roi, recall85_b)
    summary["spearman_bg_recall85"] = spearman_corr(dist_bg, recall85_b)

    quantiles = _failure_masks_from_quantiles(recall85_b, [0.1, args.failure_quantile, 0.3])
    for q, (mask, thr) in quantiles.items():
        mask = mask.astype(int)
        metrics = _stratified_cv_metrics(dist_l2.reshape(-1, 1), mask, args.risk_folds)
        if metrics is None:
            continue
        for metric_name, (mean, lo, hi) in metrics.items():
            key = f"distance_q{int(q*100)}_{metric_name}"
            summary[f"{key}_mean"] = mean
            summary[f"{key}_ci_lo"] = lo
            summary[f"{key}_ci_hi"] = hi
        summary[f"threshold_q{int(q*100)}"] = thr

    if LogisticRegression:
        failure_mask = recall85_b <= np.quantile(recall85_b, args.failure_quantile)
        feature_sets = {
            "distance": dist_l2.reshape(-1, 1),
            "roi_distance": dist_roi.reshape(-1, 1),
            "bg_distance": dist_bg.reshape(-1, 1),
            "roi_bg": np.stack([dist_roi, dist_bg], axis=1),
            "roi_aux": np.stack([dist_roi, mean_score_b, n_detections_b, contrast_b], axis=1),
        }
        for name, X in feature_sets.items():
            metrics = _stratified_cv_metrics(X, failure_mask.astype(int), args.risk_folds)
            if metrics is None:
                continue
            for metric_name, (mean, lo, hi) in metrics.items():
                summary[f"{name}_{metric_name}_mean"] = mean
                summary[f"{name}_{metric_name}_ci_lo"] = lo
                summary[f"{name}_{metric_name}_ci_hi"] = hi

    with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
        json.dump(records, f, indent=2)

    # Distance vs ΔRecall@0.85 (hexbin) + decile summary
    centers_rec, mean_rec, ci_low_rec, ci_high_rec, counts_rec = _bin_stats(dist_l2, recall85_b, bins=10)
    rho_med, rho_lo, rho_hi = _bootstrap_spearman(dist_l2, recall85_b, samples=1000)
    sorted_idx = np.argsort(dist_l2)
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(dist_l2, recall85_b, gridsize=40, cmap="Blues", mincnt=1)
    trend_x = dist_l2[sorted_idx]
    trend_y = recall85_b[sorted_idx]
    trend_label = "Trend"
    if IsotonicRegression:
        ir = IsotonicRegression(out_of_bounds="clip")
        low_p = np.quantile(trend_x, 0.05)
        high_p = np.quantile(trend_x, 0.95)
        mask_valid = (trend_x >= low_p) & (trend_x <= high_p)
        if mask_valid.sum() >= 3:
            trend_x_iso = trend_x[mask_valid]
            trend_y_iso = trend_y[mask_valid]
            trend_y = ir.fit_transform(trend_x_iso, trend_y_iso)
            trend_x = trend_x_iso
        trend_label = "Isotonic trend"
    else:
        x_lin = np.linspace(trend_x.min(), trend_x.max(), 200)
        trend_label = "Monotonic interpolation"
        trend_x = x_lin
        trend_y = np.interp(x_lin, dist_l2[sorted_idx], recall85_b[sorted_idx])
    ax.plot(trend_x, trend_y, color="tab:orange", linewidth=2, label=trend_label)
    if centers_rec.size > 0:
        ax.plot(centers_rec, mean_rec, marker="o", color="tab:green", label="Binned mean")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Layouts per bin")
    ax.set_xlabel("Embedding L2 distance")
    ax.set_ylabel("Recall@0.85 (scanned)")
    ax.set_title("Distance vs Recall@0.85")
    ax.text(
        0.05,
        0.95,
        f"Spearman ρ={rho_med:.2f} (95% CI [{rho_lo:.2f}, {rho_hi:.2f}])\nN={len(dist_l2)}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "distance_recall_hexbin.png"), dpi=200)
    plt.close(fig)

    # Decile summary (mean ± 95% CI) for Recall@0.85
    fig, ax = plt.subplots(figsize=(6, 4))
    if centers_rec.size > 0:
        ax.plot(centers_rec, mean_rec, marker="o", color="tab:orange", label="Recall@0.85")
        ax.fill_between(centers_rec, ci_low_rec, ci_high_rec, alpha=0.2, color="tab:orange", label="95% CI")
    ax.set_xlabel("Embedding L2 distance")
    ax.set_ylabel("Mean Recall@0.85 (scanned)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Recall@0.85 vs embedding distance (deciles)")
    average_bin = int(np.round(np.nanmean(counts_rec))) if counts_rec.size else 0
    ax.text(
        0.05,
        0.02,
        f"≈{average_bin} layouts per bin",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    median_dist = float(np.median(dist_l2))
    ax.axvline(median_dist, color="0.4", linestyle="--", linewidth=1, label=f"Median distance ({median_dist:.2f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "distance_binned_recall.png"), dpi=200)
    plt.close(fig)

    # Failure probability curve (logistic fit for low recall)
    failure_thr = float(np.quantile(recall85_b, args.failure_quantile))
    failure_mask = recall85_b <= failure_thr
    summary["failure_threshold"] = failure_thr
    centers_fail, failure_rate = _bin_failure_rate(dist_l2, failure_mask, bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    if centers_fail.size > 0:
        ax.plot(
            centers_fail,
            failure_rate,
            marker="o",
            color="tab:red",
            label=f"P(Recall@0.85 ≤ {failure_thr:.2f})",
        )

    cv_auc = summary.get("distance_auc_mean")
    cv_auc_lo = summary.get("distance_auc_ci_lo")
    cv_auc_hi = summary.get("distance_auc_ci_hi")

    if LogisticRegression and failure_mask.sum() > 0 and failure_mask.sum() < len(failure_mask):
        logit_line_x = np.linspace(dist_l2.min(), dist_l2.max(), 200)
        logit = LogisticRegression(solver="lbfgs", max_iter=1000)
        logit.fit(dist_l2.reshape(-1, 1), failure_mask.astype(int))
        logit_line_y = logit.predict_proba(logit_line_x.reshape(-1, 1))[:, 1]
        ax.plot(logit_line_x, logit_line_y, color="tab:purple", label="Logistic fit (visual)")

    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Embedding L2 distance")
    ax.set_ylabel("Probability of low Recall@0.85")
    ax.set_title("Low Recall risk vs distance")
    ax.grid(alpha=0.3)
    if cv_auc is not None and cv_auc_lo is not None and cv_auc_hi is not None:
        ax.text(
            0.05,
            0.02,
            f"CV AUC={cv_auc:.2f} (95% CI [{cv_auc_lo:.2f}, {cv_auc_hi:.2f}])",
            transform=ax.transAxes,
            va="bottom",
            fontsize=9,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "distance_failure_rate.png"), dpi=200)
    plt.close(fig)
    summary["auc_failure_distance"] = cv_auc

    # Pair gallery for low/high distance samples
    sorted_idx = np.argsort(dist_l2)
    low_indices = sorted_idx[: min(8, len(sorted_idx))]
    high_indices = sorted_idx[-min(8, len(sorted_idx)) :][::-1]
    _save_pair_gallery(
        records,
        low_indices,
        args.image_a,
        args.image_b,
        args.out_dir,
        "pairs_low_distance",
        "Top low-distance pairs",
    )
    _save_pair_gallery(
        records,
        high_indices,
        args.image_a,
        args.image_b,
        args.out_dir,
        "pairs_high_distance",
        "Top high-distance pairs",
    )

    # PCA panels for low/high degradation
    rng = np.random.default_rng(42)
    low_thr = float(np.quantile(delta_ap85, PCA_LOW_Q))
    high_thr = float(np.quantile(delta_ap85, PCA_HIGH_Q))
    low_mask = delta_ap85 <= low_thr
    high_mask = delta_ap85 >= high_thr
    low_fallback = delta_ap85 <= float(np.quantile(delta_ap85, 0.25))
    high_fallback = delta_ap85 >= float(np.quantile(delta_ap85, 0.75))
    low_mask = _ensure_min_mask(low_mask, low_fallback, PCA_MIN_SAMPLES)
    high_mask = _ensure_min_mask(high_mask, high_fallback, PCA_MIN_SAMPLES)
    low_idx = _sample_indices(low_mask, PCA_MAX_POINTS, rng)
    high_idx = _sample_indices(high_mask, PCA_MAX_POINTS, rng)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _plot_pca_panel(
        axs[0],
        emb_a[low_idx],
        emb_b[low_idx],
        delta_ap85[low_idx],
        f"ΔAP@0.85 ≤ q{int(PCA_LOW_Q * 100)} ({low_thr:.2f})",
        cmap="viridis",
        show_colorbar=True,
    )
    _plot_pca_panel(
        axs[1],
        emb_a[high_idx],
        emb_b[high_idx],
        delta_ap85[high_idx],
        f"ΔAP@0.85 ≥ q{int(PCA_HIGH_Q * 100)} ({high_thr:.2f})",
        cmap="coolwarm",
        show_colorbar=True,
    )
    fig.suptitle("PCA of paired embeddings (low vs high degradation)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "pca_pairs_panels.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
