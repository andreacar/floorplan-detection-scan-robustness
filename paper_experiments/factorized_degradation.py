from utils.paper_io import figure_path, table_path
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from paper_experiments.common import (  # noqa: E402
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


LEVELS = ["mild", "medium", "strong"]


def _add_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    if sigma <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    arr += rng.normal(0.0, sigma, arr.shape)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _degrade_blur(img: Image.Image, level: str, rng: np.random.Generator) -> Image.Image:
    radius = {"mild": 1.0, "medium": 2.0, "strong": 3.5}[level]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _degrade_thicken(img: Image.Image, level: str, rng: np.random.Generator) -> Image.Image:
    size = {"mild": 3, "medium": 5, "strong": 7}[level]
    sigma = {"mild": 3.0, "medium": 6.0, "strong": 10.0}[level]
    g = img.convert("L")
    g = _add_noise(g, sigma=sigma, rng=rng)
    g = g.filter(ImageFilter.MinFilter(size=size))
    return Image.merge("RGB", (g, g, g))


def _degrade_texture(img: Image.Image, level: str, rng: np.random.Generator) -> Image.Image:
    alpha = {"mild": 0.10, "medium": 0.20, "strong": 0.30}[level]
    contrast = {"mild": 0.95, "medium": 0.85, "strong": 0.75}[level]
    brightness = {"mild": 0.02, "medium": 0.04, "strong": 0.06}[level]

    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    low_h = max(8, h // 32)
    low_w = max(8, w // 32)

    low_noise = rng.normal(0.0, 1.0, (low_h, low_w)).astype(np.float32)
    noise_img = Image.fromarray(low_noise)
    noise_img = noise_img.resize((w, h), resample=Image.BILINEAR)
    noise = np.array(noise_img, dtype=np.float32)
    noise = noise / (np.std(noise) + 1e-6)

    shade = 1.0 + alpha * noise[..., None]
    arr = arr * shade
    arr = (arr - 0.5) * contrast + 0.5 + brightness
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _degrade_clutter(img: Image.Image, level: str, rng: np.random.Generator) -> Image.Image:
    cfg = {
        "mild": {"lines": 15, "rects": 8, "text": 6},
        "medium": {"lines": 35, "rects": 18, "text": 16},
        "strong": {"lines": 60, "rects": 30, "text": 28},
    }[level]

    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    w, h = out.size
    color = {"mild": (120, 120, 120), "medium": (90, 90, 90), "strong": (70, 70, 70)}[level]

    for _ in range(cfg["lines"]):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        width = int(rng.integers(1, 3 if level == "mild" else 4))
        draw.line((x1, y1, x2, y2), fill=color, width=width)

    for _ in range(cfg["rects"]):
        x1, y1 = rng.integers(0, w - 10), rng.integers(0, h - 10)
        x2 = min(w, x1 + int(rng.integers(20, 120)))
        y2 = min(h, y1 + int(rng.integers(10, 60)))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=1)

    words = ["DIM", "A1", "A2", "100", "250", "R3", "N", "S", "E", "W"]
    for _ in range(cfg["text"]):
        x, y = rng.integers(0, w - 10), rng.integers(0, h - 10)
        txt = rng.choice(words)
        draw.text((x, y), txt, fill=color, font=font)

    return out


DEGRADATIONS = {
    "blur": _degrade_blur,
    "thicken": _degrade_thicken,
    "texture": _degrade_texture,
    "clutter": _degrade_clutter,
}


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


def _compute_size_summary(areas: List[float], success50: List[float], success85: List[float]) -> Dict:
    areas_np = np.array(areas, dtype=float)
    log_area = np.log(np.maximum(areas_np, 1e-6))
    s50 = np.array(success50, dtype=float)
    s85 = np.array(success85, dtype=float)

    w50, b50 = _fit_logistic(log_area, s50)
    w85, b85 = _fit_logistic(log_area, s85)
    area50 = float(np.exp(-b50 / w50)) if abs(w50) > 1e-9 else float("nan")
    area85 = float(np.exp(-b85 / w85)) if abs(w85) > 1e-9 else float("nan")

    return {
        "count": int(len(areas)),
        "logistic_50": {"w": w50, "b": b50, "area_at_50pct": area50},
        "logistic_85": {"w": w85, "b": b85, "area_at_50pct": area85},
    }


def _evaluate_level(
    name: str,
    folders: List[str],
    image_name: str,
    model,
    processor,
    device: torch.device,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    map_raw_to_l2,
    per_class_cap: Dict[str, int],
    degrade_fn,
    rng: np.random.Generator,
    score_thresh: float,
    topk_pre: int,
    final_k: int,
    use_per_class_thresh: bool,
    clean_embeddings: Dict[str, np.ndarray],
    with_embedding: bool,
) -> Dict:
    ap50_list = []
    ap75_list = []
    missed = 0
    loose = 0
    tight = 0
    total = 0
    areas = []
    success50 = []
    success85 = []
    dist_list = []
    delta_ap50_list = []
    delta_ap75_list = []

    for folder in tqdm(folders, desc=f"{name}", leave=False):
        img_path = os.path.join(folder, image_name)
        graph_path = os.path.join(folder, "graph.json")
        if not (os.path.exists(img_path) and os.path.exists(graph_path)):
            continue

        img = load_image(img_path)
        gt_boxes, gt_labels, gt_areas = load_gt_from_graph(
            graph_path, img.size[0], img.size[1], map_raw_to_l2, label2id
        )
        if not gt_boxes:
            continue

        degraded = degrade_fn(img, rng)

        boxes, scores, labels = infer_predictions(
            model,
            processor,
            degraded,
            device,
            id2label,
            score_thresh=score_thresh,
            topk_pre=topk_pre,
            final_k=final_k,
            per_class_cap=per_class_cap,
            use_per_class_thresh=use_per_class_thresh,
        )

        ap50 = compute_ap_for_image(
            gt_boxes,
            gt_labels,
            boxes.tolist(),
            scores.tolist(),
            labels.tolist(),
            iou_thr=0.5,
            class_ids=label2id.values(),
        )
        ap75 = compute_ap_for_image(
            gt_boxes,
            gt_labels,
            boxes.tolist(),
            scores.tolist(),
            labels.tolist(),
            iou_thr=0.75,
            class_ids=label2id.values(),
        )
        ap50_list.append(ap50)
        ap75_list.append(ap75)

        matched_iou = match_greedy_by_class(
            gt_boxes,
            gt_labels,
            boxes.tolist(),
            scores.tolist(),
            labels.tolist(),
            iou_thr=0.5,
        )
        for area, iou_val in zip(gt_areas, matched_iou):
            total += 1
            areas.append(float(area))
            if iou_val < 0.5:
                missed += 1
                success50.append(0.0)
                success85.append(0.0)
            else:
                success50.append(1.0)
                if iou_val < 0.85:
                    loose += 1
                    success85.append(0.0)
                else:
                    tight += 1
                    success85.append(1.0)

        if with_embedding:
            key = os.path.basename(folder.rstrip("/"))
            clean_emb = clean_embeddings.get(key)
            if clean_emb is not None:
                emb = extract_embedding(model, processor, degraded, device)
                dist = float(np.linalg.norm(clean_emb - emb))
                dist_list.append(dist)
                delta_ap50_list.append(ap50)
                delta_ap75_list.append(ap75)

    error = {
        "total": total,
        "missed": missed,
        "loose": loose,
        "tight": tight,
        "missed_frac": missed / total if total else 0.0,
        "loose_frac": loose / total if total else 0.0,
        "tight_frac": tight / total if total else 0.0,
    }
    size = _compute_size_summary(areas, success50, success85)

    out = {
        "count_images": int(len(ap50_list)),
        "ap50_mean": float(np.mean(ap50_list)) if ap50_list else 0.0,
        "ap75_mean": float(np.mean(ap75_list)) if ap75_list else 0.0,
        "error_decomposition": error,
        "size": size,
    }

    if with_embedding and dist_list:
        dist_arr = np.array(dist_list)
        d50 = np.array(delta_ap50_list)
        d75 = np.array(delta_ap75_list)
        out["embedding"] = {
            "mean_distance": float(np.mean(dist_arr)),
            "pearson_dist_ap50": pearson_corr(dist_arr, d50),
            "pearson_dist_ap75": pearson_corr(dist_arr, d75),
            "spearman_dist_ap50": spearman_corr(dist_arr, d50),
            "spearman_dist_ap75": spearman_corr(dist_arr, d75),
        }

    return out


def _make_degrade_fn(factor: str, level: str):
    if factor == "clean":
        return lambda img, rng: img
    fn = DEGRADATIONS[factor]
    return lambda img, rng: fn(img, level, rng)


def main():
    parser = argparse.ArgumentParser(description="Factorized degradation study.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--image-name", default="model_baked.png", help="Clean image name.")
    parser.add_argument("--test-txt", default=None, help="Override TEST_TXT path.")
    parser.add_argument("--out-dir", default="paper_experiments/out/factorized", help="Output dir.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--factors",
        default="blur,thicken,texture,clutter",
        help="Comma-separated list of factors.",
    )
    parser.add_argument(
        "--levels",
        default="mild,medium,strong",
        help="Comma-separated list of levels.",
    )
    parser.add_argument("--with-embedding", action="store_true", help="Compute embedding distance.")
    parser.add_argument("--score-thresh", type=float, default=0.0, help="Score threshold.")
    parser.add_argument("--topk-pre", type=int, default=0, help="Pre top-k by score.")
    parser.add_argument("--final-k", type=int, default=0, help="Final per-image cap.")
    parser.add_argument("--use-per-class-thresh", action="store_true", help="Use per-class thresholds.")

    args = parser.parse_args()

    safe_makedirs(args.out_dir)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    factors = [f.strip() for f in args.factors.split(",") if f.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    for level in levels:
        if level not in LEVELS:
            raise ValueError(f"Unsupported level: {level}")

    classes, label2id, id2label, map_raw_to_l2 = load_label_maps()
    processor, model = load_model(args.ckpt, device)
    test_dirs = load_test_dirs(args.test_txt)
    if args.limit and args.limit > 0:
        test_dirs = test_dirs[: args.limit]

    per_class_cap = DEFAULT_PER_CLASS_CAP.copy()

    clean_embeddings = {}
    if args.with_embedding:
        print("Computing clean embeddings...")
        for folder in tqdm(test_dirs, desc="Clean embeddings"):
            img_path = os.path.join(folder, args.image_name)
            if not os.path.exists(img_path):
                continue
            img = load_image(img_path)
            emb = extract_embedding(model, processor, img, device)
            key = os.path.basename(folder.rstrip("/"))
            clean_embeddings[key] = emb

    start = time.time()
    results = {
        "config": {
            "ckpt": args.ckpt,
            "image_name": args.image_name,
            "test_txt": args.test_txt,
            "factors": factors,
            "levels": levels,
            "score_thresh": args.score_thresh,
            "topk_pre": args.topk_pre,
            "final_k": args.final_k,
            "use_per_class_thresh": args.use_per_class_thresh,
            "with_embedding": args.with_embedding,
            "limit": args.limit,
        },
        "baseline": {},
        "factors": {},
    }

    baseline = _evaluate_level(
        name="clean",
        folders=test_dirs,
        image_name=args.image_name,
        model=model,
        processor=processor,
        device=device,
        label2id=label2id,
        id2label=id2label,
        map_raw_to_l2=map_raw_to_l2,
        per_class_cap=per_class_cap,
        degrade_fn=_make_degrade_fn("clean", "mild"),
        rng=rng,
        score_thresh=args.score_thresh,
        topk_pre=args.topk_pre,
        final_k=args.final_k,
        use_per_class_thresh=args.use_per_class_thresh,
        clean_embeddings=clean_embeddings,
        with_embedding=False,
    )
    results["baseline"] = baseline

    for factor in factors:
        if factor not in DEGRADATIONS:
            raise ValueError(f"Unknown factor: {factor}")
        results["factors"][factor] = {}
        for level in levels:
            name = f"{factor}:{level}"
            degrade_fn = _make_degrade_fn(factor, level)
            out = _evaluate_level(
                name=name,
                folders=test_dirs,
                image_name=args.image_name,
                model=model,
                processor=processor,
                device=device,
                label2id=label2id,
                id2label=id2label,
                map_raw_to_l2=map_raw_to_l2,
                per_class_cap=per_class_cap,
                degrade_fn=degrade_fn,
                rng=rng,
                score_thresh=args.score_thresh,
                topk_pre=args.topk_pre,
                final_k=args.final_k,
                use_per_class_thresh=args.use_per_class_thresh,
                clean_embeddings=clean_embeddings,
                with_embedding=args.with_embedding,
            )
            out["delta_ap50_mean"] = baseline["ap50_mean"] - out["ap50_mean"]
            out["delta_ap75_mean"] = baseline["ap75_mean"] - out["ap75_mean"]
            results["factors"][factor][level] = out

    results["elapsed_sec"] = float(time.time() - start)

    out_path = os.path.join(args.out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
